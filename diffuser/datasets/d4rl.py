import collections
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import gym
import numpy as np


@contextmanager
def suppress_output():
    """
    A context manager that redirects stdout and stderr to devnull
    https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_output():
    # d4rl prints out a variety of warnings
    import d4rl  # noqa: F401


class MultiAgentEnvWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action[0])
        obs_n = np.array([obs])
        reward_n = np.array([reward])
        done_n = np.array([done])
        return obs_n, reward_n, done_n, info

    def reset(self):
        obs = self.env.reset()
        obs_n = np.array([obs])
        return obs_n


# -----------------------------------------------------------------------------#
# -------------------------------- general api --------------------------------#
# -----------------------------------------------------------------------------#


def load_environment(name):
    if type(name) != str:
        # name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name

    if hasattr(env, "metadata"):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
    env.metadata["global_feats"] = []
    return MultiAgentEnvWrapper(env)


def get_dataset(env):
    dataset = env.get_dataset()
    return dataset


def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """

    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1

        for k in dataset:
            if "metadata" in k:
                continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if "maze2d" in env.name:
                episode_data = process_maze2d_episode(episode_data)
            episode_data = pretend_multiagent(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


def pretend_multiagent(episode_data):
    for k in episode_data:
        episode_data[k] = np.expand_dims(episode_data[k], 1)
    return episode_data


# -----------------------------------------------------------------------------#
# -------------------------------- maze2d fixes -------------------------------#
# -----------------------------------------------------------------------------#


def process_maze2d_episode(episode):
    """
    adds in `next_observations` field to episode
    """
    assert "next_observations" not in episode
    next_observations = episode["observations"][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode["next_observations"] = next_observations
    return episode
