import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from og_marl.environments import smac, mamujoco


def main(env_name: str, map_name: str, quality: str):
    add_agent_id_to_obs = True
    dataset_dir = Path(f"diffuser/datasets/data/{env_name}/{map_name}/{quality}")

    file_path = Path(dataset_dir)
    sub_dir_to_idx = {}
    idx = 0
    for subdir in os.listdir(file_path):
        if file_path.joinpath(subdir).is_dir():
            sub_dir_to_idx[subdir] = idx
            idx += 1

    def get_fname_idx(file_name):
        if env_name == "smac":
            dir_idx = sub_dir_to_idx[file_name.split("/")[-2]] * 1000
            return dir_idx + int(file_name.split("log_")[-1].split(".")[0])
        elif env_name == "mamujoco":
            dir_idx = sub_dir_to_idx[file_name.split("/")[-2]] * 1000
            return dir_idx + int(file_name.split("/")[-1].split("log_")[-1].split(".")[0])
        else:
            raise ValueError(f"Unknown environment {env_name}")

    filenames = [str(file_name) for file_name in file_path.glob("**/*.tfrecord")]
    filenames = sorted(filenames, key=get_fname_idx)

    if env_name == "smac":
        env = smac.SMAC(map_name)
    elif env_name == "mamujoco":
        env = mamujoco.Mujoco(map_name)
    else:
        raise ValueError(f"Unknown environment {env_name}")
    agents = env.agents

    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    raw_dataset = filename_dataset.flat_map(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").map(
            env._decode_fn
        )
    )

    period = 10

    # Split the dataset into multiple batches
    batch_size = 2048
    databatches = raw_dataset.batch(batch_size)

    (
        all_observations,
        all_actions,
        all_rewards,
        all_discounts,
        all_logprobs,
    ) = ([], [], [], [], [])
    if env_name == "smac":
        all_states, all_legals = [], []
    all_path_lengths = []

    path_length = 0
    (
        path_observations,
        path_actions,
        path_rewards,
        path_logprobs,
        path_discounts,
    ) = ([], [], [], [], [])
    if env_name == "smac":
        path_states, path_legals = [], []
    for databatch in tqdm(databatches):
        extras = databatch.extras
        batch_size = len(extras["zero_padding_mask"])
        if env_name == "smac":
            states = extras["s_t"]

        observations, actions, rewards, discounts, logprobs = ([], [], [], [], [])
        if env_name == "smac":
            legals = []
        for agent in agents:
            observations.append(databatch.observations[agent].observation.numpy())
            if env_name == "smac":
                legals.append(databatch.observations[agent].legal_actions.numpy())
            actions.append(databatch.actions[agent].numpy())
            rewards.append(databatch.rewards[agent].numpy())
            discounts.append(databatch.discounts[agent].numpy())
            if "logprobs" in extras:
                logprobs.append(extras["logprobs"][agent].numpy())

        observations = np.stack(observations, axis=2)
        if env_name == "smac":
            legals = np.stack(legals, axis=2)
        actions = np.stack(actions, axis=2)
        rewards = np.stack(rewards, axis=-1)
        discounts = np.stack(discounts, axis=-1)
        if "logprobs" in extras:
            logprobs = np.stack(logprobs, axis=2)

        for idx in range(batch_size):
            zero_padding_mask = extras["zero_padding_mask"][idx][:period]
            path_length += np.sum(zero_padding_mask, dtype=int)

            if env_name == "smac":
                path_states.append(states[idx, :period])
                path_legals.append(legals[idx, :period])
            path_observations.append(observations[idx, :period])
            path_actions.append(actions[idx, :period])
            path_rewards.append(rewards[idx, :period])
            path_discounts.append(discounts[idx, :period])
            if "logprobs" in extras:
                path_logprobs.append(logprobs[idx, :period])

            if (
                int(path_discounts[-1][-1, 0]) == 0
                or path_length >= env.max_episode_length
            ):
                path_observations = np.concatenate(path_observations, axis=0)
                if add_agent_id_to_obs:
                    T, N = path_observations.shape[:2]
                    agent_ids = []
                    for i in range(N):
                        agent_id = tf.one_hot(i, depth=N)
                        agent_ids.append(agent_id)
                    agent_ids = tf.stack(agent_ids, axis=0)

                    # Repeat along time dim
                    agent_ids = tf.stack([agent_ids] * T, axis=0)
                    agent_ids = agent_ids.numpy()

                    path_observations = np.concatenate(
                        [agent_ids, path_observations], axis=-1
                    )

                if env_name == "smac":
                    all_states.append(np.concatenate(path_states, axis=0)[:path_length])
                    all_legals.append(np.concatenate(path_legals, axis=0)[:path_length])
                all_observations.append(path_observations[:path_length])
                all_actions.append(np.concatenate(path_actions, axis=0)[:path_length])
                all_rewards.append(np.concatenate(path_rewards, axis=0)[:path_length])
                all_discounts.append(
                    np.concatenate(path_discounts, axis=0)[:path_length]
                )
                if "logprobs" in extras:
                    all_logprobs.append(
                        np.concatenate(path_logprobs, axis=0)[:path_length]
                    )
                all_path_lengths.append(path_length)

                (
                    path_observations,
                    path_actions,
                    path_rewards,
                    path_discounts,
                    path_logprobs,
                ) = ([], [], [], [], [])
                if env_name == "smac":
                    path_states, path_legals = [], []
                path_length = 0

    """ Concatenate Episodes """
    if env_name == "smac":
        all_states = np.concatenate(all_states, axis=0)
        all_legals = np.concatenate(all_legals, axis=0)
    all_observations = np.concatenate(all_observations, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_rewards = np.concatenate(all_rewards, axis=0)
    all_discounts = np.concatenate(all_discounts, axis=0)
    if "logprobs" in extras:
        all_logprobs = np.concatenate(all_logprobs, axis=0)
    all_path_lengths = np.array(all_path_lengths)

    """ Save Numpy Arrays """
    if env_name == "smac":
        np.save(
            f"diffuser/datasets/data/{env_name}/{map_name}/{quality}/states.npy",
            all_states,
        )
        np.save(
            f"diffuser/datasets/data/{env_name}/{map_name}/{quality}/legals.npy",
            all_legals,
        )
    np.save(
        f"diffuser/datasets/data/{env_name}/{map_name}/{quality}/obs.npy",
        all_observations,
    )
    np.save(
        f"diffuser/datasets/data/{env_name}/{map_name}/{quality}/actions.npy",
        all_actions,
    )
    np.save(
        f"diffuser/datasets/data/{env_name}/{map_name}/{quality}/rewards.npy",
        all_rewards,
    )
    np.save(
        f"diffuser/datasets/data/{env_name}/{map_name}/{quality}/discounts.npy",
        all_discounts,
    )
    if "logprobs" in extras:
        np.save(
            f"diffuser/datasets/data/{env_name}/{map_name}/{quality}/logprobs.npy",
            all_logprobs,
        )
    np.save(
        f"diffuser/datasets/data/{env_name}/{map_name}/{quality}/path_lengths.npy",
        all_path_lengths,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="smac")
    parser.add_argument("--map_name", type=str, default="3m")
    parser.add_argument("--quality", type=str, default="Good")
    args = parser.parse_args()

    main(args.env_name, args.map_name, args.quality)
