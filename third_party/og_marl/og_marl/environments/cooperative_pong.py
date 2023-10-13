"""Wraper for Cooperative Pong."""
from typing import Any, Dict, List, Union

import dm_env
import numpy as np
from acme import specs
from pettingzoo.butterfly import cooperative_pong_v5
import supersuit

from mava import types
from mava.utils.wrapper_utils import parameterized_restart
from mava.wrappers.env_wrappers import ParallelEnvWrapper

from og_marl.environments.pettingzoo_base import PettingZooBase


class CooperativePong(PettingZooBase):
    """Environment wrapper for PettingZoo MARL environments."""

    def __init__(
        self,
    ):
        """Constructor for parallel PZ wrapper.

        Args:
            environment (ParallelEnv): parallel PZ env.
            env_preprocess_wrappers (Optional[List], optional): Wrappers
                that preprocess envs.
                Format (env_preprocessor, dict_with_preprocessor_params).
            return_state_info: return extra state info
        """
        self._environment = cooperative_pong_v5.parallel_env(render_mode="rgb_array")
        # Wrap environment with supersuit pre-process wrappers
        self._environment = supersuit.color_reduction_v0(self._environment, mode="R")
        self._environment = supersuit.resize_v1(
            self._environment, x_size=145, y_size=84
        )
        self._environment = supersuit.dtype_v0(self._environment, dtype="float32")
        self._environment = supersuit.normalize_obs_v0(self._environment)

        self._agents = self._environment.possible_agents
        self._reset_next_step = True
        self._done = False

    def _create_state_representation(self, observations):
        if self._step_type == dm_env.StepType.FIRST:
            self._state_history = np.zeros((84, 145, 4), "float32")

        state = np.expand_dims(observations["paddle_0"][:, :], axis=-1)

        # framestacking
        self._state_history = np.concatenate(
            (state, self._state_history[:, :, :3]), axis=-1
        )

        return self._state_history

    def _add_zero_obs_for_missing_agent(self, observations):
        for agent in self._agents:
            if agent not in observations:
                observations[agent] = np.zeros((84, 145), "float32")
        return observations

    def _convert_observations(
        self, observations: List, done: bool
    ) -> types.Observation:
        """Convert SMAC observation so it's dm_env compatible.

        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.

        Returns:
            types.Observation: dm compatible observations.
        """
        olt_observations = {}
        for i, agent in enumerate(self._agents):

            if agent == "paddle_0":
                agent_obs = observations[agent][:, :110]  # hide the other agent
            else:
                agent_obs = observations[agent][:, 35:]  # hide the other agent

            agent_obs = np.expand_dims(agent_obs, axis=-1)
            olt_observations[agent] = types.OLT(
                observation=agent_obs,
                legal_actions=np.ones(3, "float32"),  # three actions in pong, all legal
                terminal=np.asarray(done, dtype="float32"),
            )

        return olt_observations

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns extra spec (format) of the env.

        Returns:
            Dict[str, specs.BoundedArray]: extra spec.
        """
        state_spec = {"s_t": np.zeros((84, 145, 4), "float32")}  # four stacked frames

        return state_spec

    def observation_spec(self) -> Dict[str, types.OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        observation_specs = {}
        for agent in self._agents:

            obs = np.zeros((84, 110, 1), "float32")

            observation_specs[agent] = types.OLT(
                observation=obs,
                legal_actions=np.ones(3, "float32"),
                terminal=np.asarray(True, "float32"),
            )

        return observation_specs

    def action_spec(
        self,
    ) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Action spec.

        Returns:
            spec for actions.
        """
        action_specs = {}
        for agent in self._agents:
            action_specs[agent] = specs.DiscreteArray(
                num_values=3, dtype="int64"  # three actions
            )
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Reward spec.

        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        reward_specs = {}
        for agent in self._agents:
            reward_specs[agent] = np.array(1, "float32")
        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Discount spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_specs = {}
        for agent in self._agents:
            discount_specs[agent] = np.array(1, "float32")
        return discount_specs

    @property
    def agents(self) -> List:
        """Agents still alive in env (not done).

        Returns:
            List: alive agents in env.
        """
        return self._agents

    @property
    def possible_agents(self) -> List:
        """All possible agents in env.

        Returns:
            List: all possible agents in env.
        """
        return self._agents

    @property
    def environment(self):
        """Returns the wrapped environment.

        Returns:
            ParallelEnv: parallel env.
        """
        return self._environment

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
