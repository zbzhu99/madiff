import pytest
import numpy as np

from diffuser.datasets.smac import SMAC


@pytest.mark.parametrize("map_name", ["3m"])
def test_smac_env(map_name):
    env = SMAC(map_name)
    obs = env.reset()
    assert len(obs.shape) == 2 and obs.shape[0] == env.num_agents

    legal_actions = env.get_legal_actions()
    acts = np.array(
        [np.random.choice(np.where(legal_act > 0)[0]) for legal_act in legal_actions]
    )

    next_obs, rews, dones, infos = env.step(acts)
    assert next_obs.shape == obs.shape
    assert "battle_won" in infos
