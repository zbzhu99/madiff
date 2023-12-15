import importlib
from typing import Callable, List, Optional

import numpy as np
import torch

from diffuser.datasets.buffer import ReplayBuffer
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.datasets.preprocessing import get_preprocess_fn


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env_type: str = "d4rl",
        env: str = "hopper-medium-replay",
        n_agents: int = 2,
        horizon: int = 64,
        normalizer: str = "LimitsNormalizer",
        preprocess_fns: List[Callable] = [],
        use_action: bool = True,
        discrete_action: bool = False,
        max_path_length: int = 1000,
        max_n_episodes: int = 10000,
        termination_penalty: float = 0,
        use_padding: bool = True,  # when path_length is smaller than max_path_length
        discount: float = 0.99,
        returns_scale: float = 400.0,
        include_returns: bool = False,
        include_env_ts: bool = False,
        use_state: bool = False,
        history_horizon: int = 0,
        agent_share_parameters: bool = False,
        use_seed_dataset: bool = False,
        decentralized_execution: bool = False,
        use_inverse_dynamic: bool = True,
        seed: int = None,
    ):
        if use_seed_dataset:
            assert (
                env_type == "mpe"
            ), f"Seed dataset only supported for MPE, not {env_type}"

        env_mod_name = {
            "d4rl": "diffuser.datasets.d4rl",
            "mahalfcheetah": "diffuser.datasets.mahalfcheetah",
            "mamujoco": "diffuser.datasets.mamujoco",
            "mpe": "diffuser.datasets.mpe",
            "smac": "diffuser.datasets.smac_env",
            "smacv2": "diffuser.datasets.smacv2_env",
        }[env_type]
        env_mod = importlib.import_module(env_mod_name)

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = env_mod.load_environment(env)
        self.global_feats = env.metadata["global_feats"]

        self.use_inverse_dynamic = use_inverse_dynamic
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
        self.include_env_ts = include_env_ts
        self.use_state = use_state
        self.decentralized_execution = decentralized_execution

        if env_type == "mpe":
            if use_seed_dataset:
                itr = env_mod.sequence_dataset(env, self.preprocess_fn, seed=seed)
            else:
                itr = env_mod.sequence_dataset(env, self.preprocess_fn)
        elif env_type == "smac" or env_type == "smacv2":
            itr = env_mod.sequence_dataset(env, self.preprocess_fn, use_state=use_state)
        else:
            itr = env_mod.sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(
            n_agents,
            max_n_episodes,
            max_path_length,
            termination_penalty,
            global_feats=self.global_feats,
        )
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(
            fields,
            normalizer,
            path_lengths=fields["path_lengths"],
            agent_share_parameters=agent_share_parameters,
            global_feats=self.global_feats,
        )

        self.state_dim = fields.states.shape[-1] if self.use_state else 0
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1] if self.use_action else 0
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths

        self.indices = self.make_indices(fields.path_lengths)

        if self.discrete_action:
            # smac has discrete actions, so we only need to normalize observations
            self.normalize(
                keys=["states", "observations"] if self.use_state else ["observations"]
            )
        else:
            self.normalize()

        self.pad_future()
        if self.history_horizon > 0:
            self.pad_history()

        print(fields)

    def pad_future(self, keys=None):
        if keys is None:
            keys = ["normed_observations", "rewards"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if self.use_action:
                if self.discrete_action:
                    keys.append("actions")
                else:
                    keys.append("normed_actions")

            if self.use_state:
                keys.append("normed_states")

        for key in keys:
            shape = self.fields[key].shape
            self.fields[key] = np.concatenate(
                [
                    self.fields[key],
                    np.zeros(
                        (shape[0], self.horizon - 1, *shape[2:]),
                        dtype=self.fields[key].dtype,
                    ),
                ],
                axis=1,
            )

    def pad_history(self, keys=None):
        if keys is None:
            keys = ["normed_observations", "rewards"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if self.use_action:
                if self.discrete_action:
                    keys.append("actions")
                else:
                    keys.append("normed_actions")

            if self.use_state:
                keys.append("normed_states")

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

    def normalize(self, keys: List[str] = None):
        """
        normalize fields that will be predicted by the diffusion model
        """
        if keys is None:
            keys = ["observations", "actions"] if self.use_action else ["observations"]
            if self.use_state:
                keys.append("states")

        for key in keys:
            shape = self.fields[key].shape
            array = self.fields[key].reshape(shape[0] * shape[1], *shape[2:])
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(shape)

    def make_indices(self, path_lengths):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """

        indices = []
        for i, path_length in enumerate(path_lengths):
            if self.use_padding:
                max_start = path_length - 1
            else:
                max_start = path_length - self.horizon
                if max_start < 0:
                    continue

            # get `end` and `mask_end` for each `start`
            for start in range(max_start):
                end = start + self.horizon
                mask_end = min(end, path_length)
                indices.append((i, start, end, mask_end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations, agent_idx: Optional[int] = None):
        """
        condition on current observations for planning
        """

        ret_dict = {}
        if self.decentralized_execution:
            cond_observations = np.zeros_like(observations[: self.history_horizon + 1])
            cond_observations[:, agent_idx] = observations[
                : self.history_horizon + 1, agent_idx
            ]
            ret_dict["agent_idx"] = torch.LongTensor([[agent_idx]])
        else:
            cond_observations = observations[: self.history_horizon + 1]
        ret_dict[(0, self.history_horizon + 1)] = cond_observations
        return ret_dict

    def __len__(self):
        if self.decentralized_execution:
            return len(self.indices) * self.n_agents
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        if self.decentralized_execution:
            path_ind, start, end, mask_end = self.indices[idx // self.n_agents]
            agent_idx = idx % self.n_agents
        else:
            path_ind, start, end, mask_end = self.indices[idx]
            agent_idx = None

        # shift by `self.history_horizon`
        history_start = start
        start = history_start + self.history_horizon
        end = end + self.history_horizon
        mask_end = mask_end + self.history_horizon

        observations = self.fields.normed_observations[path_ind, history_start:end]
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, history_start:end]
            else:
                actions = self.fields.normed_actions[path_ind, history_start:end]
        if self.use_state:
            states = self.fields.normed_states[path_ind, history_start:end]

        loss_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        loss_masks[self.history_horizon : mask_end - history_start] = 1.0
        if self.use_inverse_dynamic:
            if self.decentralized_execution:
                loss_masks[self.history_horizon, agent_idx] = 0.0
            else:
                loss_masks[self.history_horizon] = 0.0

        attention_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        attention_masks[self.history_horizon : mask_end - history_start] = 1.0
        if self.decentralized_execution:
            attention_masks[: self.history_horizon, agent_idx] = 1.0
        else:
            attention_masks[: self.history_horizon] = 1.0

        conditions = self.get_conditions(observations, agent_idx)
        if self.use_action:
            trajectories = np.concatenate([actions, observations], axis=-1)
        else:
            trajectories = observations

        batch = {
            "x": trajectories,
            "cond": conditions,
            "loss_masks": loss_masks,
            "attention_masks": attention_masks,
        }

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start : -self.horizon + 1]
            discounts = self.discounts[: len(rewards)]
            returns = (discounts * rewards).sum(axis=0).squeeze(-1)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch["returns"] = returns

        if self.include_env_ts:
            env_ts = (
                np.arange(history_start, start + self.horizon) - self.history_horizon
            )
            env_ts[np.where(env_ts < 0)] = self.max_path_length
            env_ts[np.where(env_ts >= self.max_path_length)] = self.max_path_length
            batch["env_ts"] = env_ts

        if self.use_state:
            batch["states"] = states

        if "legal_actions" in self.fields.keys:
            batch["legal_actions"] = self.fields.legal_actions[
                path_ind, history_start:end
            ]

        return batch


class ValueDataset(SequenceDataset):
    """
    adds a value field to the datapoints for training the value function
    """

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.include_returns == True

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        value_batch = {
            "x": batch["x"],
            "cond": batch["cond"],
            "returns": batch["returns"].mean(axis=-1),
        }
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

        batch = {"observations": observations, "actions": actions}
        return batch
