import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from og_marl.offline_tools.offline_dataset import MAOfflineDataset
from og_marl.environments.smac import SMAC


def main(map_name: str, quality: str):
    add_agent_id_to_obs = True

    file_path = Path(f"diffuser/datasets/data/smac/{map_name}/{quality}")

    env = SMAC(map_name)
    agents = env.agents

    dataset = MAOfflineDataset(
        environment=env,
        logdir=file_path,
        batch_size=128,
        shuffle_buffer_size=5000,
    )

    raw_dataset = dataset.filename_dataset.flat_map(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").map(
            dataset._decode_fn
        )
    )

    period = 10

    # 将数据集分为多个 batch
    batch_size = 2048
    databatches = raw_dataset.batch(batch_size)

    all_observations, all_legals, all_actions, all_rewards, all_discounts, all_logprobs = [], [], [], [], [], []
    all_path_lengths = []

    path_length = 0
    path_observations, path_legals, path_actions, path_rewards, path_logprobs, path_discounts = [], [], [], [], [], []
    for databatch in tqdm(databatches):
        extras = databatch.data.extras
        batch_size = len(extras["zero_padding_mask"])

        observations, legals, actions, rewards, discounts, logprobs = [], [], [], [], [], []
        for agent in agents:

            observations.append(databatch.data.observations[agent].observation.numpy())
            legals.append(databatch.data.observations[agent].legal_actions.numpy())
            actions.append(databatch.data.actions[agent].numpy())
            rewards.append(databatch.data.rewards[agent].numpy())
            discounts.append(databatch.data.discounts[agent].numpy())
            if "logprobs" in extras:
                logprobs.append(extras["logprobs"][agent].numpy())

        observations = np.stack(observations, axis=2)
        legals = np.stack(legals, axis=2)
        actions = np.stack(actions, axis=2)
        rewards = np.stack(rewards, axis=-1)
        discounts = np.stack(discounts, axis=-1)
        if "logprobs" in extras:
            logprobs = np.stack(logprobs, axis=2)

        for idx in range(batch_size):
            zero_padding_mask = extras["zero_padding_mask"][idx][:period]
            path_length += np.sum(zero_padding_mask, dtype=int)

            path_observations.append(observations[idx, :period])
            path_legals.append(legals[idx, :period])
            path_actions.append(actions[idx, :period])
            path_rewards.append(rewards[idx, :period])
            path_discounts.append(discounts[idx, :period])
            if "logprobs" in extras:
                path_logprobs.append(logprobs[idx, :period])

            if int(path_discounts[-1][-1, 0]) == 0 or path_length >= env.max_episode_length:
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

                    path_observations = np.concatenate([agent_ids, path_observations], axis=-1)

                all_observations.append(path_observations[:path_length])
                all_legals.append(np.concatenate(path_legals, axis=0)[:path_length])
                all_actions.append(np.concatenate(path_actions, axis=0)[:path_length])
                all_rewards.append(np.concatenate(path_rewards, axis=0)[:path_length])
                all_discounts.append(np.concatenate(path_discounts, axis=0)[:path_length])
                if "logprobs" in extras:
                    all_logprobs.append(np.concatenate(path_logprobs, axis=0)[:path_length])
                all_path_lengths.append(path_length)
                path_observations, path_legals, path_actions, path_rewards, path_discounts, path_logprobs = [], [], [], [], [], []
                path_length = 0

    """ Concatenate Episodes """
    all_observations = np.concatenate(all_observations, axis=0)
    all_legals = np.concatenate(all_legals, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_rewards = np.concatenate(all_rewards, axis=0)
    all_discounts = np.concatenate(all_discounts, axis=0)
    if "logprobs" in extras:
        all_logprobs = np.concatenate(all_logprobs, axis=0)
    all_path_lengths = np.array(all_path_lengths)

    """ Save Numpy Arrays """
    np.save(f"diffuser/datasets/data/smac/{map_name}/{quality}/obs.npy", all_observations)
    np.save(f"diffuser/datasets/data/smac/{map_name}/{quality}/legals.npy", all_legals)
    np.save(f"diffuser/datasets/data/smac/{map_name}/{quality}/actions.npy", all_actions)
    np.save(f"diffuser/datasets/data/smac/{map_name}/{quality}/rewards.npy", all_rewards)
    np.save(f"diffuser/datasets/data/smac/{map_name}/{quality}/discounts.npy", all_discounts)
    if "logprobs" in extras:
        np.save(f"diffuser/datasets/data/smac/{map_name}/{quality}/logprobs.npy", all_logprobs)
    np.save(f"diffuser/datasets/data/smac/{map_name}/{quality}/path_lengths.npy", all_path_lengths)


if __name__ == "__main__":
    # map_names = ["3m", "5m_vs_6m"]
    # qualities = ["Good", "Medium", "Poor"]
    map_names = ["3m"]
    qualities = ["Good"]
    for map_name in map_names:
        for quality in qualities:
            main(map_name, quality)
