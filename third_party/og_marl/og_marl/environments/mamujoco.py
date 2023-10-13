"""Wraper for Multi-agent Mujoco."""
from typing import Any, Dict, List, Optional, Union
import dm_env
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec

from multiagent_mujoco.mujoco_multi import MujocoMulti
from mava import types
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
from mava.wrappers.env_wrappers import ParallelEnvWrapper

def get_mamujoco_args(scenario):
    env_args = {
        "agent_obsk": 1,
        "episode_limit": 1000,
        "global_categories": "qvel,qpos",
    }
    if scenario.lower() == "4ant":
        env_args["scenario"] = "Ant-v2"
        env_args["agent_conf"] = "4x2"
    elif scenario.lower() == "2ant":
        env_args["scenario"] = "Ant-v2"
        env_args["agent_conf"] = "2x4"
    elif scenario.lower() == "2halfcheetah":
        env_args["scenario"] = "HalfCheetah-v2"
        env_args["agent_conf"] = "2x3"
    return env_args

class Mujoco(ParallelEnvWrapper):
    """Environment wrapper for PettingZoo MARL environments."""

    def __init__(self, env_name, discrete=False, num_discrete_bins=10):
        """Constructor for parallel PZ wrapper.

        Args:
            environment (ParallelEnv): parallel PZ env.
            env_preprocess_wrappers (Optional[List], optional): Wrappers
                that preprocess envs.
                Format (env_preprocessor, dict_with_preprocessor_params).
            return_state_info: return extra state info
        """
        env_args = get_mamujoco_args(env_name)
        self._environment = MujocoMulti(env_args=env_args)
        self._agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self.num_agents = len(self._agents)
        self.num_actions = self._environment.n_actions
        self._reset_next_step = True
        self._done = False
        self.max_episode_length = self._environment.episode_limit
        self._discrete = discrete
        self._num_discrete_bins = num_discrete_bins
        self._discrete_actions = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    self._discrete_actions.append([float(i), float(j), float(k)])
        self._num_discrete_actions = len(self._discrete_actions)

    def reset(self) -> dm_env.TimeStep:
        """Resets the env.

        Returns:
            dm_env.TimeStep: dm timestep.
        """
        # Reset the environment
        self._environment.reset()
        self._done = False

        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

        # Get observation from env
        observation = self.environment.get_obs()
        legal_actions = self._get_legal_actions()
        observations = self._convert_observations(
            observation, legal_actions, self._done
        )

        # Set env discount to 1 for all agents
        discount_spec = self.discount_spec()
        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self._agents
        }

        # Set reward to zero for all agents
        rewards = {agent: np.array(0, "float32") for agent in self._agents}

        # State info
        state = self._environment.get_state().astype("float32")
        extras = {"s_t": state}

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

        # Convert dict of actions to list for Mujoco
        # mujoco_actions = list(actions.values())

        # if self._discrete:
        #     mujoco_actions = [self._discrete_actions[action] for action in mujoco_actions]

        mujoco_actions = []
        for agent in self._agents:
            mujoco_actions.append(actions[agent])

        # Step the Mujoco environment
        reward, self._done, self._info = self._environment.step(mujoco_actions)

        # Get the next observations
        next_observations = self._environment.get_obs()
        legal_actions = self._get_legal_actions()
        next_observations = self._convert_observations(
            next_observations, legal_actions, self._done
        )

        # Convert team reward to agent-wise rewards
        rewards = {agent: np.array(reward, "float32") for agent in self.agents}

        # State info
        state = self._environment.get_state().astype("float32")
        extras = {"s_t": state}

        if self._done:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True

            # Discount on last timestep set to zero
            self._discounts = {
                agent: convert_np_type(self.discount_spec()[agent].dtype, 0.0)
                for agent in self._agents
            }
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

    def env_done(self) -> bool:
        """Check if env is done.

        Returns:
            bool: bool indicating if env is done.
        """
        return self._done

    def _get_legal_actions(self) -> List:
        """Get legal actions from the environment."""
        legal_actions = []
        for i, _ in enumerate(self._agents):
            if self._discrete:
                legal_actions.append(np.ones(self._num_discrete_actions, "float32"))
            else:
                legal_actions.append(
                    np.array(
                        self._environment.get_avail_agent_actions(i), dtype="float32"
                    )
                )
        return legal_actions

    def _convert_observations(
        self, observations: List, legal_actions: List, done: bool
    ) -> types.Observation:
        """Convert Mujoco observation so it's dm_env compatible.

        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.

        Returns:
            types.Observation: dm compatible observations.
        """
        olt_observations = {}
        for i, agent in enumerate(self._agents):

            observation = observations[i].astype(np.float32)
            olt_observations[agent] = types.OLT(
                observation=observation,
                legal_actions=legal_actions[i],
                terminal=np.asarray([done], dtype=np.float32),
            )

        return olt_observations

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns extra spec (format) of the env.

        Returns:
            Dict[str, specs.BoundedArray]: extra spec.
        """
        return {"s_t": self._environment.get_state().astype("float32")}

    def observation_spec(self) -> Dict[str, types.OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        observation_spec = np.zeros(self._environment.get_obs_size(), "float32")
        if self._discrete:
            legal_actions_spec = np.zeros(self._num_discrete_actions, "float32")
        else:
            legal_actions_spec = np.zeros(self.num_actions, "float32")

        observation_specs = {}
        for i, agent in enumerate(self._agents):

            observation_specs[agent] = types.OLT(
                observation=observation_spec,
                legal_actions=legal_actions_spec,
                terminal=np.asarray([True], dtype=np.float32),
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
        for i, agent in enumerate(self._agents):
            if self._discrete:
                action_specs[agent] = np.array(1, int)
            else:
                action_specs[agent] = _convert_to_spec(
                    self._environment.action_space[i]
                )
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Reward spec.

        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        reward_specs = {}
        for agent in self._agents:
            reward_specs[agent] = specs.Array((), np.float32)
        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Discount spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_specs = {}
        for agent in self._agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def get_stats(self) -> Optional[Dict]:
        """Return extra stats to be logged.

        Returns:
            extra stats to be logged.
        """
        return self._environment.get_stats()

    def render(self, mode):
        # Error when rendering. OPENGL init problem
        return self._environment.env.render(mode)

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
    def environment(self) -> MujocoMulti:
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
