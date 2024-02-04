import numpy as np
from einops import rearrange, repeat


class DummyMaskGenerator:
    def __call__(self, shape):
        mask = np.ones(shape, dtype=bool)
        return mask


class MultiAgentMaskGenerator:
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        # obs mask setup
        history_horizon: int = 10,
        # action mask
        action_visible: bool = False,
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.history_horizon = history_horizon
        self.action_visible = action_visible

    def __call__(self, shape: tuple, agent_mask: np.ndarray):
        if len(shape) == 4:
            B, T, _, D = shape  # b t a f
        else:
            B = None
            T, _, D = shape  # t a f
        if self.action_visible:
            assert D == (self.action_dim + self.observation_dim)
        else:
            assert D == self.observation_dim

        # generate obs mask
        steps = np.arange(0, T)
        obs_mask = np.tile(
            (steps < self.history_horizon + 1).reshape(T, 1), (1, self.observation_dim)
        )

        # generate action mask
        if self.action_visible:
            action_mask = np.tile((steps < self.history_horizon).reshape(T, 1), (1, D))

        visible_mask = obs_mask
        if self.action_visible:
            visible_mask = np.concatenate([action_mask, visible_mask], dim=-1)

        # the history of invisible agents are conditioned to be always zero
        invisible_mask = np.tile((steps < self.history_horizon).reshape(T, 1), (1, D))
        # agent_mask[a_idx] = True if agent a_idx is visible -> mask[a_idx] = visible_mask
        mask = np.stack([invisible_mask, visible_mask], axis=0)[agent_mask.astype(int)]
        mask = rearrange(mask, "a t f -> t a f")
        if B is not None:
            mask = repeat(mask, "t a f -> b t a f", b=B)

        return mask
