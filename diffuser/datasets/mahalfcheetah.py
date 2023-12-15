import collections
import os

import gym
import numpy as np
from multiagent_mujoco.mujoco_multi import MujocoMulti

ADD_AGENT_ID = False


class ObsAgentIDWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.one_hot_agent_ids = []
        for i in range(self.n_agents):
            agent_id = np.eye(self.n_agents)[i]
            self.one_hot_agent_ids.append(agent_id)
        self.one_hot_agent_ids = np.stack(self.one_hot_agent_ids, axis=0)

        self.observation_space = [
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=((obs_space.shape[0] + self.n_agents,)),
            )
            for obs_space in self.observation_space
        ]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate([self.one_hot_agent_ids, obs], axis=1)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = np.concatenate([self.one_hot_agent_ids, obs], axis=1)
        return obs


def load_environment(name):
    if type(name) != str:
        # name is already an environment
        return name

    idx = name.find(
        "-", name.find("-") + 1
    )  # the second '-' divides the env name and data split
    env_name, data_split = name[:idx], name[idx + 1 :]

    if env_name == "HalfCheetah-v2":
        env_kwargs = dict(
            agent_conf="2x3",
            agent_obsk=0,
            episode_limit=1000,
        )
    else:
        raise NotImplementedError(
            f"Multi-agent Mujoco environment {env_name} not supported."
        )

    env = MujocoMulti(env_args={"scenario": env_name, **env_kwargs})
    if hasattr(env, "metadata"):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
    env.metadata["data_split"] = data_split
    env.metadata["name"] = env_name
    env.metadata["global_feats"] = []
    if ADD_AGENT_ID:
        env = ObsAgentIDWrapper(env)
    return env


def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An MultiAgentEnv object.
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

    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "data/mahalfcheetah",
        env.scenario,
        env.metadata["data_split"],
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory not found: {}".format(dataset_path))

    n_agents = env.n_agents
    for idx, seed_dir in enumerate(os.listdir(dataset_path)):
        seed_path = os.path.join(dataset_path, seed_dir)
        if not os.path.isdir(seed_path):
            continue

        observations = np.stack(
            [
                np.load(os.path.join(seed_path, "obs_{}.npy".format(agent_idx)))
                for agent_idx in range(n_agents)
            ],
            axis=1,
        )
        actions = np.stack(
            [
                np.load(os.path.join(seed_path, "acs_{}.npy".format(agent_idx)))
                for agent_idx in range(n_agents)
            ],
            axis=1,
        )
        rewards = np.stack(
            [
                np.load(os.path.join(seed_path, "rews_{}.npy".format(agent_idx)))
                for agent_idx in range(n_agents)
            ],
            axis=1,
        )
        dones = np.stack(
            [
                np.load(os.path.join(seed_path, "dones_{}.npy".format(agent_idx)))
                for agent_idx in range(n_agents)
            ],
            axis=1,
        )

        if ADD_AGENT_ID:
            observations = np.concatenate(
                (
                    np.tile(env.one_hot_agent_ids[None], (observations.shape[0], 1, 1)),
                    observations,
                ),
                axis=2,
            )

        data_ = collections.defaultdict(list)
        for obs, act, rew, done in zip(observations, actions, rewards, dones):
            data_["observations"].append(obs)
            data_["actions"].append(act)
            data_["rewards"].append(rew)
            data_["terminals"].append(done)

            if done.all():
                data_["timeouts"] = np.zeros_like(data_["terminals"])
                if len(data_["observations"]) == env.episode_limit:
                    data_["terminals"][-1][:] = 0.0
                    data_["timeouts"][-1][:] = 1.0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                yield episode_data
                data_ = collections.defaultdict(list)
