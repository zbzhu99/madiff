import numpy as np
import torch

from diffuser.datasets.nba import load_environment
from diffuser.datasets.sequence import NBASequenceDataset
from diffuser.models.nba_ma_temporal import (
    PlayerConvAttentionDeconv,
    PlayerSharedConvAttentionTemporalValue,
)


def test_nba_env():
    env = load_environment("train")
    obs = env.reset()
    obs, reward, done, _ = env.step(0)
    assert (
        obs.shape == (64, 10, 4)
        and reward.shape == (64, 10, 1)
        and done.shape == (64, 10, 1)
    )


def test_sequence_dataset():
    dataset = NBASequenceDataset(
        env_type="nba",
        env="test",
        n_agents=10,
        use_action=False,
        max_path_length=20000,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
    )

    for batch in dataloader:
        player_idxs = batch[0][..., 0]
        player_hoop_sides = batch[0][..., -1]
        assert (
            np.unique(player_idxs, axis=1).shape[1] == 1
        ), "More than 10 players in a sequence."
        assert (
            np.unique(player_hoop_sides, axis=1).shape[1] == 1
        ), "Switch hoop sides in a sequence."


def test_value():
    batch_size, horizon = 32, 64
    value_func = PlayerSharedConvAttentionTemporalValue(
        horizon=horizon, transition_dim=2, cond_dim=1, n_agents=10
    ).to("cpu")
    x = torch.randint(0, 20, (batch_size, horizon, 10, 4)).to("cpu") / 2
    t = torch.randint(0, 100, (batch_size,)).to("cpu")
    pred_value = value_func(x, t)
    assert pred_value.shape == (
        batch_size,
        1,
    ), f"Wrong value shape {pred_value.shape}, {(batch_size, 1)} expected."


def test_diffusion_model():
    batch_size, horizon = 32, 64
    model = PlayerConvAttentionDeconv(
        horizon=horizon, transition_dim=2, cond_dim=1, n_agents=10
    ).to("cpu")
    x = torch.randint(0, 20, (batch_size, horizon, 10, 4)).to("cpu") / 2
    t = torch.randint(0, 100, (batch_size,)).to("cpu")

    model_output = model(x, t)
    assert model_output.shape == (
        batch_size,
        horizon,
        10,
        2,
    ), f"Wrong diffusion shape {model_output.shape}, {(batch_size, horizon, 10, 2)} expected."
