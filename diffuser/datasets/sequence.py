from collections import namedtuple

import numpy as np
import torch

from .buffer import ReplayBuffer
from .normalization import DatasetNormalizer
from .preprocessing import get_preprocess_fn

RewardBatch = namedtuple("Batch", "trajectories conditions masks returns")
Batch = namedtuple("Batch", "trajectories conditions masks")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")
BCBatch = namedtuple("BCBatch", "observations actions")


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env_type="d4rl",
        env="hopper-medium-replay",
        n_agents=2,
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        use_action=True,
        discrete_action=False,
        max_path_length=1000,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=True,
        discount=0.99,
        returns_scale=1000,
        include_returns=False,
        abs_pos=False,
        history_horizon=0,
        mask_others_history=True,
        agent_share_parameters=False,
    ):
        assert (
            history_horizon == 0
        ), f"History horizon {history_horizon} not supported for SequenceDataset. Use HistoryCondSequenceDataset instead"

        if env_type == "d4rl":
            from .d4rl import load_environment, sequence_dataset
        elif env_type == "ma_mujoco":
            from .ma_mujoco import load_environment, sequence_dataset

            assert preprocess_fns == [], "MA Mujoco does not support preprocessing"
        elif env_type == "mpe":
            from .mpe import load_environment, sequence_dataset

            assert preprocess_fns == [], "MPE does not support preprocessing"
        elif env_type == "smac":
            from .smac import load_environment, sequence_dataset

            assert preprocess_fns == [], "SMAC does not support preprocessing"
        elif env_type == "smac-mat":
            from .smac_mat import load_environment, sequence_dataset

            assert preprocess_fns == [], "SMAC does not support preprocessing"
        elif env_type == "nba":
            from .nba import load_environment, sequence_dataset

            assert preprocess_fns == [], "NBA does not support preprocessing"
        else:
            raise NotImplementedError(env_type)

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = (
            load_environment(env)
            if env_type != "nba"
            else load_environment(env, nba_hz=self.nba_hz)
        )

        self.returns_scale = returns_scale
        self.n_agents = n_agents
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None, None]
        self.use_padding = use_padding
        self.use_action = use_action
        self.discrete_action = discrete_action
        self.include_returns = include_returns
        if env_type == "mpe":
            itr = sequence_dataset(env, self.preprocess_fn, abs_pos=abs_pos)
        elif env_type == "nba":
            itr = sequence_dataset(env, self.preprocess_fn, mode=env.metadata["mode"])
        else:
            itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(
            n_agents, max_n_episodes, max_path_length, termination_penalty
        )
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields,
            normalizer,
            path_lengths=fields["path_lengths"],
            agent_share_parameters=agent_share_parameters,
        )

        if env_type == "nba":
            self.indices = self.nba_make_indices(
                fields.path_lengths,
                fields.player_idxs,
                fields.player_hoop_sides,
                horizon,
                test_partially=False if self.env.metadata["mode"] == "train" else True,
            )
        else:
            self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        if self.use_action:
            self.action_dim = fields.actions.shape[-1]
        else:
            self.action_dim = 0
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths

        if self.discrete_action:
            # smac has discrete actions, so we only need to normalize observations
            self.normalize(keys=["observations"])
        elif env_type != "nba" or env != "test":
            self.normalize()
        else:
            print(
                "NBA evaluation doesn't need normalizer, use training normalizer instead"
            )

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=None):
        """
        normalize fields that will be predicted by the diffusion model
        """
        if keys is None:
            keys = ["observations", "actions"] if self.use_action else ["observations"]

        for key in keys:
            array = self.fields[key].reshape(
                self.n_episodes * self.max_path_length, self.n_agents, -1
            )
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(
                self.n_episodes, self.max_path_length, self.n_agents, -1
            )

    def make_indices(self, path_lengths, horizon):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """

        indices = []
        for i, path_length in enumerate(path_lengths):
            # max_start = min(path_length - 1, self.max_path_length - horizon)
            max_start = min(path_length - 1, self.max_path_length - 1)
            if not self.use_padding:
                max_start = min(max_start, path_length - 1)
            for start in range(max_start):
                end = start + horizon
                if not self.use_padding:
                    mask_end = min(start + horizon, path_length)
                else:
                    mask_end = min(start + horizon, self.max_path_length)
                indices.append((i, start, end, mask_end))
        indices = np.array(indices)
        return indices

    def nba_make_indices(
        self,
        path_lengths,
        player_idxs,
        player_hoop_sides,
        horizon,
        test_partially=False,
    ):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """
        N = self.nba_eval_valid_samples
        partially_samps_per_gameid = int(np.ceil(N / len(path_lengths)))
        indices = []
        for i, path_length in enumerate(path_lengths):
            # max_start = min(path_length - 1, self.max_path_length - horizon)
            consistent = False
            max_start = min(path_length - 1, self.max_path_length - 1)
            if not self.use_padding:
                max_start = min(max_start, path_length - 1)
            gaps = max_start // partially_samps_per_gameid
            tot_indeces = (
                range(max_start)
                if test_partially == False
                else range(0, max_start, gaps)
            )
            for start in tot_indeces:
                end = start + horizon
                if not self.use_padding:
                    mask_end = min(start + horizon, path_length)
                else:
                    mask_end = min(start + horizon, self.max_path_length)
                # do not return sequences with player substitution or change of hoop side
                # NOTE: NOT sure whether change hoop side should be returned
                if consistent == False:
                    if (
                        len(np.unique(player_idxs[i, start:mask_end])) == 10
                        and np.unique(
                            player_hoop_sides[i, start:mask_end], axis=0
                        ).shape[0]
                        == 1
                    ):
                        consistent = True
                        indices.append((i, start, end, mask_end))
                else:
                    if np.all(
                        player_idxs[i, mask_end - 2] == player_idxs[i, mask_end - 1]
                    ) and np.all(
                        player_hoop_sides[i, mask_end - 1]
                        == player_hoop_sides[i, mask_end - 2]
                    ):
                        indices.append((i, start, end, mask_end))
                    else:
                        consistent = False
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        """
        condition on current observation for planning
        """

        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end, mask_end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, start:end]
            else:
                actions = self.fields.normed_actions[path_ind, start:end]

        if mask_end < end:
            observations = np.concatenate(
                [
                    observations,
                    np.zeros(
                        (end - mask_end, self.n_agents, observations.shape[-1]),
                        dtype=observations.dtype,
                    ),
                ],
                axis=0,
            )
            if self.use_action:
                actions = np.concatenate(
                    [
                        actions,
                        np.zeros(
                            (end - mask_end, self.n_agents, actions.shape[-1]),
                            dtype=actions.dtype,
                        ),
                    ],
                    axis=0,
                )

        masks = np.zeros((observations.shape[0], observations.shape[1]))
        masks[: mask_end - start] = 1.0

        conditions = self.get_conditions(observations)
        if self.use_action:
            trajectories = np.concatenate([actions, observations], axis=-1)
        else:
            trajectories = observations

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum(axis=0).squeeze(-1)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, masks, returns)
        else:
            batch = Batch(trajectories, conditions, masks)

        return batch


class CTDESequenceDataset(SequenceDataset):
    def get_conditions(self, observations, agent_idx):
        """
        condition on current observation for planning
        """

        return {
            0: observations[0, agent_idx],
            "agent_idx": torch.LongTensor([agent_idx]),
        }

    def __len__(self):
        return len(self.indices) * self.n_agents

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end, mask_end = self.indices[idx // self.n_agents]

        observations = self.fields.normed_observations[path_ind, start:end]
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, start:end]
            else:
                actions = self.fields.normed_actions[path_ind, start:end]

        if mask_end < end:
            observations = np.concatenate(
                [
                    observations,
                    np.zeros(
                        (end - mask_end, self.n_agents, observations.shape[-1]),
                        dtype=observations.dtype,
                    ),
                ],
                axis=0,
            )
            if self.use_action:
                actions = np.concatenate(
                    [
                        actions,
                        np.zeros(
                            (end - mask_end, self.n_agents, actions.shape[-1]),
                            dtype=actions.dtype,
                        ),
                    ],
                    axis=0,
                )

        masks = np.zeros((observations.shape[0], observations.shape[1]))
        masks[:mask_end] = 1.0

        conditions = self.get_conditions(observations, idx % self.n_agents)
        if self.use_action:
            trajectories = np.concatenate([actions, observations], axis=-1)
        else:
            trajectories = observations

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum(axis=0).squeeze(-1)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, masks, returns)
        else:
            batch = Batch(trajectories, conditions, masks)

        return batch


class NBASequenceDataset(SequenceDataset):
    def __init__(
        self, nba_hz=5, nba_eval_valid_samples=1000, *args, **kwargs,
    ):
        self.nba_hz = nba_hz
        self.nba_eval_valid_samples = nba_eval_valid_samples
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx, eps=1e-4):
        assert self.use_action == False
        path_ind, start, end, mask_end = self.indices[idx]

        player_idxs = self.fields.player_idxs[path_ind, start:end]
        observations = self.fields.normed_observations[path_ind, start:end]
        player_hoop_sides = self.fields.player_hoop_sides[path_ind, start:end]

        if mask_end < end:
            observations = np.concatenate(
                [
                    observations,
                    np.zeros(
                        (end - mask_end, self.n_agents, observations.shape[-1]),
                        dtype=observations.dtype,
                    ),
                ],
                axis=0,
            )

        masks = np.zeros((observations.shape[0], observations.shape[1]))
        masks[:mask_end] = 1.0
        conditions = self.get_conditions(observations)
        for key, val in list(conditions.items()):
            conditions[key] = val[:, 1:-1] if val.shape[-1] == 4 else val
        conditions["player_idxs"] = player_idxs
        conditions["player_hoop_sides"] = player_hoop_sides

        batch = Batch(observations, conditions, masks)
        return batch


class HistoryCondSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env_type="d4rl",
        env="hopper-medium-replay",
        n_agents=2,
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        use_action=True,
        discrete_action=False,
        max_path_length=1000,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=True,
        discount=0.99,
        returns_scale=1000,
        include_returns=False,
        history_horizon=4,
        agent_share_parameters=False,
    ):
        assert (
            history_horizon > 0
        ), f"history_horizon {history_horizon} must be larger than zero, otherwise use SequenceDataset"

        if env_type == "d4rl":
            from .d4rl import load_environment, sequence_dataset
        elif env_type == "ma_mujoco":
            from .ma_mujoco import load_environment, sequence_dataset

            assert preprocess_fns == [], "MA Mujoco does not support preprocessing"
        elif env_type == "mpe":
            from .mpe import load_environment, sequence_dataset

            assert preprocess_fns == [], "MPE does not support preprocessing"
        elif env_type == "smac":
            from .smac import load_environment, sequence_dataset

            assert preprocess_fns == [], "SMAC does not support preprocessing"
        elif env_type == "smac-mat":
            from .smac_mat import load_environment, sequence_dataset

            assert preprocess_fns == [], "SMAC does not support preprocessing"
        elif env_type == "nba":
            from .nba import load_environment, sequence_dataset

            assert preprocess_fns == [], "NBA does not support preprocessing"

        else:
            raise NotImplementedError(env_type)

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = (
            load_environment(env)
            if env_type != "nba"
            else load_environment(env, nba_hz=self.nba_hz)
        )

        self.returns_scale = returns_scale
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None, None]
        self.use_padding = use_padding
        self.use_action = use_action
        self.discrete_action = discrete_action
        self.include_returns = include_returns

        if env_type == "nba":
            itr = sequence_dataset(env, self.preprocess_fn, mode=env.metadata["mode"])
        else:
            itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(
            n_agents, max_n_episodes, max_path_length, termination_penalty
        )
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields,
            normalizer,
            path_lengths=fields["path_lengths"],
            agent_share_parameters=agent_share_parameters,
        )

        self.observation_dim = fields.observations.shape[-1]
        if self.use_action:
            self.action_dim = fields.actions.shape[-1]
        else:
            self.action_dim = 0
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        if self.discrete_action:
            # smac has discrete actions, so we only need to normalize observations
            self.normalize(keys=["observations"])
        elif env_type != "nba" or env != "test":
            self.normalize()
        else:
            print(
                "NBA evaluation doesn't need normalizer, use training normalizer instead"
            )

        if env_type != "nba":  # nba dosen't need to padding forehead
            if self.discrete_action:
                self.pad_history(keys=["normed_observations", "actions"])
            else:
                self.pad_history()

        if env_type == "nba":
            self.indices = self.nba_make_indices(
                fields.path_lengths,
                fields.player_idxs,
                fields.player_hoop_sides,
                horizon,
                history_horizon,
                test_partially=False if self.env.metadata["mode"] == "train" else True,
            )
        else:
            self.indices = self.make_indices(fields.path_lengths, horizon, history_horizon)

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def pad_history(self, keys=None):
        if keys is None:
            keys = (
                ["normed_observations", "normed_actions"]
                if self.use_action
                else ["normed_observations"]
            )

        for key in keys:
            shape = self.fields[key].shape
            self.fields[key] = np.concatenate(
                [
                    np.zeros(
                        (shape[0], self.history_horizon, *shape[2:]),
                        dtype=self.fields[key].dtype,
                    ),
                    self.fields[key],
                ],
                axis=1,
            )

    def normalize(self, keys=None):
        """
        normalize fields that will be predicted by the diffusion model
        """

        if keys is None:
            keys = ["observations", "actions"] if self.use_action else ["observations"]

        for key in keys:
            array = self.fields[key].reshape(
                self.n_episodes * self.max_path_length, self.n_agents, -1
            )
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(
                self.n_episodes, self.max_path_length, self.n_agents, -1
            )

    def make_indices(self, path_lengths, horizon, history_horizon):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """

        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - 1)
            if not self.use_padding:
                max_start = min(max_start, path_length - 1)
            for start in range(max_start):
                end = start + horizon
                if not self.use_padding:
                    mask_end = min(start + horizon, path_length)
                else:
                    mask_end = min(start + horizon, self.max_path_length)
                indices.append(
                    (
                        i,
                        start,  # start of history
                        start
                        + history_horizon,  # start of prediction (w. current) / end of history
                        end + history_horizon,  # end of prediction
                        mask_end + history_horizon,  # end of mask
                    )
                )
        indices = np.array(indices)
        return indices

    def nba_make_indices(
        self,
        path_lengths,
        player_idxs,
        player_hoop_sides,
        horizon,
        history_horizon,
        test_partially=False,
    ):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """
        N = self.nba_eval_valid_samples
        partially_samps_per_gameid = int(np.ceil(N / len(path_lengths)))
        indices = []
        for i, path_length in enumerate(path_lengths):
            # max_start = min(path_length - 1, self.max_path_length - horizon)
            consistent = False
            max_start = min(path_length - 1, self.max_path_length - 1)
            if not self.use_padding:
                max_start = min(max_start, path_length - 1)
            gaps = max_start // partially_samps_per_gameid
            tot_indeces = (
                range(max_start)
                if test_partially == False
                else range(0, max_start, gaps)
            )
            for start in tot_indeces:
                end = start + horizon
                if not self.use_padding:
                    mask_end = min(start + horizon, path_length)
                else:
                    mask_end = min(start + horizon, self.max_path_length)
                # do not return sequences with player substitution or change of hoop side
                # NOTE: NOT sure whether change hoop side should be returned
                if consistent == False:
                    if (
                        len(
                            np.unique(
                                player_idxs[i, start : mask_end + history_horizon]
                            )
                        )
                        == 10
                        and np.unique(
                            player_hoop_sides[i, start : mask_end + history_horizon],
                            axis=0,
                        ).shape[0]
                        == 1
                    ):
                        consistent = True
                        indices.append(
                            (
                                i,
                                start,
                                start + history_horizon,
                                end + history_horizon,
                                mask_end + history_horizon,
                            )
                        )
                else:
                    if np.all(
                        player_idxs[i, mask_end + history_horizon - 2]
                        == player_idxs[i, mask_end + history_horizon - 1]
                    ) and np.all(
                        player_hoop_sides[i, mask_end + history_horizon - 1]
                        == player_hoop_sides[i, mask_end + history_horizon - 2]
                    ):
                        indices.append(
                            (
                                i,
                                start,
                                start + history_horizon,
                                end + history_horizon,
                                mask_end + history_horizon,
                            )
                        )
                    else:
                        consistent = False
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations, history_horizon):
        """
        condition on current observation for planning
        """

        return {(0, history_horizon + 1): observations[: history_horizon + 1]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, history_start, start, end, mask_end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, history_start:end]
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, history_start:end]
            else:
                actions = self.fields.normed_actions[path_ind, history_start:end]

        if mask_end < end:
            observations = np.concatenate(
                [
                    observations,
                    np.zeros(
                        (end - mask_end, self.n_agents, observations.shape[-1]),
                        dtype=observations.dtype,
                    ),
                ],
                axis=0,
            )
            if self.use_action:
                actions = np.concatenate(
                    [
                        actions,
                        np.zeros(
                            (end - mask_end, self.n_agents, actions.shape[-1]),
                            dtype=actions.dtype,
                        ),
                    ],
                    axis=0,
                )

        masks = np.zeros((observations.shape[0], observations.shape[1]))
        masks[: mask_end - history_start] = 1.0

        conditions = self.get_conditions(observations, self.history_horizon)
        if self.use_action:
            trajectories = np.concatenate([actions, observations], axis=-1)
        else:
            trajectories = observations

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum(axis=0).squeeze(-1)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, masks, returns)
        else:
            batch = Batch(trajectories, conditions, masks)

        return batch


class NBAHistoryCondSequenceDataset(HistoryCondSequenceDataset):
    def __init__(
        self, nba_hz=5, nba_eval_valid_samples=1000, *args, **kwargs,
    ):
        self.nba_hz = nba_hz
        self.nba_eval_valid_samples = nba_eval_valid_samples
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx, eps=1e-4):
        assert self.use_action == False
        path_ind, history_start, start, end, mask_end = self.indices[idx]

        player_idxs = self.fields.player_idxs[path_ind, history_start:end]
        observations = self.fields.normed_observations[path_ind, history_start:end]
        player_hoop_sides = self.fields.player_hoop_sides[path_ind, history_start:end]

        if mask_end < end:
            observations = np.concatenate(
                [
                    observations,
                    np.zeros(
                        (end - mask_end, self.n_agents, observations.shape[-1]),
                        dtype=observations.dtype,
                    ),
                ],
                axis=0,
            )

        masks = np.zeros((observations.shape[0], observations.shape[1]))
        masks[: mask_end - history_start] = 1.0
        conditions = self.get_conditions(observations, self.history_horizon)
        for key, val in list(conditions.items()):
            assert len(val.shape) == 3  # history_horizon, num_agent, num_feature
            conditions[key] = val[:, :, 1:-1] if val.shape[-1] == 4 else val
        conditions["player_idxs"] = player_idxs
        conditions["player_hoop_sides"] = player_hoop_sides

        batch = Batch(observations, conditions, masks)
        return batch


class CTDEHistoryCondSequenceDataset(HistoryCondSequenceDataset):
    def __init__(self, *args, mask_others_history=False, **kwargs):
        super().__init__(*args, **kwargs)
        # whether to predict others' history trajectories
        self.mask_others_history = mask_others_history

    def get_conditions(self, observations, history_horizon, agent_idx):
        """
        condition on current observation for planning
        """

        return {
            (0, history_horizon + 1): observations[: history_horizon + 1, agent_idx],
            "agent_idx": torch.LongTensor([[agent_idx]]),
        }

    def __len__(self):
        return len(self.indices) * self.n_agents

    def __getitem__(self, idx, eps=1e-4):
        path_ind, history_start, start, end, mask_end = self.indices[
            idx // self.n_agents
        ]
        agent_idx = idx % self.n_agents

        observations = self.fields.normed_observations[path_ind, history_start:end]
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, history_start:end]
            else:
                actions = self.fields.normed_actions[path_ind, history_start:end]

        if mask_end < end:
            observations = np.concatenate(
                [
                    observations,
                    np.zeros(
                        (end - mask_end, self.n_agents, observations.shape[-1]),
                        dtype=observations.dtype,
                    ),
                ],
                axis=0,
            )
            if self.use_action:
                actions = np.concatenate(
                    [
                        actions,
                        np.zeros(
                            (end - mask_end, self.n_agents, actions.shape[-1]),
                            dtype=actions.dtype,
                        ),
                    ],
                    axis=0,
                )

        masks = np.zeros((observations.shape[0], observations.shape[1]))
        masks[: mask_end - history_start] = 1.0

        if self.mask_others_history:
            masks[: self.history_horizon] = 0.0
            masks[: self.history_horizon, agent_idx] = 1.0

        conditions = self.get_conditions(observations, self.history_horizon, agent_idx)
        if self.use_action:
            trajectories = np.concatenate([actions, observations], axis=-1)
        else:
            trajectories = observations

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum(axis=0).squeeze(-1)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, masks, returns)
        else:
            batch = Batch(trajectories, conditions, masks)

        return batch


class GoalDataset(SequenceDataset):
    def get_conditions(self, observations):
        """
        condition on both the current observation and the last observation in the plan
        """
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    """
    adds a value field to the datapoints for training the value function
    """

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.include_returns == True

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        value_batch = ValueBatch(
            batch.trajectories, batch.conditions, batch.returns.mean(axis=-1)
        )
        return value_batch


class BCSequenceDataset(SequenceDataset):
    def __init__(
        self,
        env_type="d4rl",
        env="hopper-medium-replay",
        n_agents=2,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=10000,
        agent_share_parameters=False,
    ):
        super().__init__(
            env_type=env_type,
            env=env,
            n_agents=n_agents,
            normalizer=normalizer,
            preprocess_fns=preprocess_fns,
            max_path_length=max_path_length,
            max_n_episodes=max_n_episodes,
            agent_share_parameters=agent_share_parameters,
            horizon=1,
            use_action=True,
            termination_penalty=0.0,
            use_padding=False,
            discount=1.0,
            include_returns=False,
        )

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end, _ = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        batch = BCBatch(observations, actions)
        return batch
