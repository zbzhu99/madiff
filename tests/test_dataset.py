import torch

from diffuser.datasets import HistoryCondSequenceDataset
from diffuser.utils.training import cycle


def test_history_conditioned_dataset():
    raise NotImplementedError("This test needs to be rewritten")
    dataset = HistoryCondSequenceDataset(
        env_type="mpe",
        env="simple_spread-medium-replay",
        normalizer="CDFNormalizer",
        n_agents=3,
        horizon=3,
        history_horizon=1,
        max_path_length=25,
        max_n_episodes=200000,
        use_padding=True,
        include_returns=True,
        discount=0.99,
        termination_penalty=0.0,
    )
    dataloader = cycle(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
        )
    )

    batch = next(dataloader)

    print(batch.trajectories)


def test_smac_dataset():
    from diffuser.datasets.smac import load_environment, sequence_dataset

    env = load_environment("3m-Good")
    itr = sequence_dataset(env, [])

    for i, episode in enumerate(itr):
        pass
