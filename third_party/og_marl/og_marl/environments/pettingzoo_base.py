"""Base wraper for Cooperative Pettingzoo environments."""
from typing import Any, Dict, List, Union
import dm_env
import numpy as np
from acme import specs
import matplotlib.pyplot as plt
from acme.wrappers.gym_wrapper import _convert_to_spec
from mava import types
from mava.utils.wrapper_utils import parameterized_restart
from mava.wrappers.env_wrappers import ParallelEnvWrapper


class PettingZooBase(ParallelEnvWrapper):
    """Environment wrapper for PettingZoo MARL environments."""

    def __init__(self):
        """Constructor for parallel PZ wrapper."""
        self._environment = None
        self._agents = None

        self.num_actions = None
        self.action_dim = None
        self.max_trajectory_length = None

        self._reset_next_step = True
        self._done = False

    def reset(self) -> dm_env.TimeStep:
        """Resets the env.

        Returns:
            dm_env.TimeStep: dm timestep.
        """
        # Reset the environment
        observations = self._environment.reset()
        self._done = False
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

        # Global state
        state = self._create_state_representation(observations)
        if state is not None:
            extras = {"s_t": state}
        else:
            extras = {}

        # Convert observations to OLT format
        observations = self._convert_observations(observations, self._done)

        # Set env discount to 1 for all agents and all non-terminal timesteps
        self._discounts = {agent: np.array(1, "float32") for agent in self._agents}

        # Set reward to zero for all agents
        rewards = {agent: np.array(0, "float32") for agent in self._agents}

        return parameterized_restart(rewards, self._discounts, observations), extras

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps in env.

        Args:
            actions (Dict[str, np.ndarray]): actions per agent.

        Returns:
            dm_env.TimeStep: dm timestep
        """
        # Possibly reset the environment
        if self._reset_next_step:
            return self.reset()

        actions = self._preprocess_actions(actions)

        # Step the environment
        next_observations, pz_rewards, dones, truncated, _ = self._environment.step(
            actions
        )

        # Add zero-observations to missing agents
        next_observations = self._add_zero_obs_for_missing_agent(next_observations)

        rewards = {}
        for agent in self._agents:
            if agent in pz_rewards:
                rewards[agent] = np.array(pz_rewards[agent], "float32")
            else:
                rewards[agent] = np.array(0, "float32")

        # Set done flag
        self._done = all(dones.values()) or all(truncated.values())

        # Global state
        state = self._create_state_representation(next_observations)
        if state is not None:
            extras = {"s_t": state}
        else:
            extras = {}

        # for i in range(4):
        #     plt.imshow(state[:,:,i])
        #     plt.savefig(f"state_{i}.png")

        # Convert next observations to OLT format
        next_observations = self._convert_observations(next_observations, self._done)

        # for i, observation in enumerate(next_observations.values()):
        #     plt.imshow(observation.observation)
        #     plt.savefig(f"obs_{i}.png")

        if self._done:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True

            # Discount on last timestep set to zero
            self._discounts = {agent: np.array(0, "float32") for agent in self._agents}
        else:
            self._step_type = dm_env.StepType.MID

        # Create timestep object
        timestep = dm_env.TimeStep(
            observation=next_observations,
            reward=rewards,
            discount=self._discounts,
            step_type=self._step_type,
        )

        return timestep, extras

    def render(self, mode):
        return self._environment.render()

    def _add_zero_obs_for_missing_agent(self, observations):
        for agent in self._agents:
            if agent not in observations:
                observations[agent] = np.zeros_like(self.observation_spec()[agent].observation)
        return observations

    def _preprocess_actions(self, actions):
        return actions

    def env_done(self) -> bool:
        """Check if env is done.

        Returns:
            bool: bool indicating if env is done.
        """
        return self._done

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
        raise NotImplementedError

    def _create_state_representation(self, observations):

        raise NotImplementedError

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        raise NotImplementedError

    def observation_spec(self) -> Dict[str, types.OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        raise NotImplementedError

    def action_spec(
        self,
    ) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Action spec.

        Returns:
            spec for actions.
        """
        action_specs = {}
        for agent in self._agents:
            action_specs[agent] = _convert_to_spec(
                self._environment.action_space(agent)
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

    def get_stats(self):
        """Return extra stats to be logged.

        Returns:
            extra stats to be logged.
        """
        return {}

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
