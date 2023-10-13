import copy
import os

import gym
import numpy as np
import pygame
from gym import spaces
from multiagent.core import Agent


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world,
        reset_callback=None,
        reward_callback=None,
        observation_callback=None,
        info_callback=None,
        done_callback=None,
        post_step_callback=None,
        render_mode=None,
        discrete_action=False,
        max_timestep=25,
    ):
        # print ('\033[1;31mdiscrete_action: {}\033[1;0m'.format(discrete_action))

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.post_step_callback = post_step_callback
        # environment parameters
        self.discrete_action_space = discrete_action
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = (
            world.discrete_action if hasattr(world, "discrete_action") else False
        )
        # if true, every agent has the same reward
        self.shared_reward = False
        self.time = 0
        self.max_timestep = max_timestep

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(
                    low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,)
                )
            if agent.movable:
                total_action_space.append(u_action_space)

            # communication action space
            c_action_space = spaces.Discrete(world.dim_c)
            if not agent.silent:
                total_action_space.append(c_action_space)

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all(
                    [
                        isinstance(act_space, spaces.Discrete)
                        for act_space in total_action_space
                    ]
                ):
                    act_space = spaces.MultiDiscrete(
                        [[0, act_space.n - 1] for act_space in total_action_space]
                    )
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,))
            )
            agent.action.c = np.zeros(self.world.dim_c)

        self.render_mode = render_mode
        if self.render_mode == "rgb_array":
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )
        self.renderOn = False

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {"n": []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            # self._set_action(action_n[i], agent, self.action_space[i])
            self._set_action(
                copy.deepcopy(action_n[i]), agent, self.action_space[i]
            )  # modified by ling: 0717
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent_idx, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n["n"].append(self._get_info(agent))

            # print ('[{}] orig_pos: {}, action: {}, pos: {}'.format(agent_idx, orig_pos_list[agent_idx], action_n[agent_idx], self.agents[agent_idx].state.p_pos))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n
        if self.post_step_callback is not None:
            self.post_step_callback(self.world)

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        # print ('\033[1;32m_set_action for agent {}: {}\033[1;0m'.format(agent, action))

        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, spaces.MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index : (index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def enable_render(self, mode="human"):
        if not self.renderOn:
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self, *args):
        if self.render_mode is None:
            print("You are calling render method without specifying any render mode.")
            return

        self.enable_render(self.render_mode)

        self.draw()
        pygame.display.flip()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        if len(self.world.agents) <= 3:
            boundary = 1
        elif len(self.world.agents) <= 10:
            boundary = 2

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / boundary) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / boundary) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            if isinstance(entity, Agent):
                pygame.draw.circle(
                    self.screen, entity.color * 200, (x, y), entity.size * 200
                )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            else:
                pygame.draw.circle(
                    self.screen, entity.color * 200, (x, y), entity.size * 350
                )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
                # pygame.draw.circle(
                #     self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
                # )  # borders
            # assert (
            #     0 < x < self.width and 0 < y < self.height
            # ), f"Coordinates {(x, y)} are out of bounds."

            # text
            # if isinstance(entity, Agent):
            #     if entity.silent:
            #         continue
            #     if np.all(entity.state.c == 0):
            #         word = "_"
            #     elif self.continuous_actions:
            #         word = (
            #             "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
            #         )
            #     else:
            #         word = alphabet[np.argmax(entity.state.c)]

            #     message = entity.name + " sends " + word + "   "
            #     message_x_pos = self.width * 0.05
            #     message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
            #     self.game_font.render_to(
            #         self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
            #     )
            #     text_line += 1

    # def draw(self):
    #     # clear screen
    #     self.screen.fill((255, 255, 255))

    #     # update bounds to center around agent
    #     all_poses = [entity.state.p_pos for entity in self.world.entities]
    #     cam_range = np.max(np.abs(np.array(all_poses)))

    #     # update geometry and text positions
    #     text_line = 0
    #     for e, entity in enumerate(self.world.entities):
    #         # geometry
    #         x, y = entity.state.p_pos
    #         y *= (
    #             -1
    #         )  # this makes the display mimic the old pyglet setup (ie. flips image)
    #         x = (
    #             (x / cam_range) * self.width // 2 * 0.9
    #         )  # the .9 is just to keep entities from appearing "too" out-of-bounds
    #         y = (y / cam_range) * self.height // 2 * 0.9
    #         x += self.width // 2
    #         y += self.height // 2
    #         pygame.draw.circle(
    #             self.screen, entity.color * 200, (x, y), entity.size * 350
    #         )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
    #         pygame.draw.circle(
    #             self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
    #         )  # borders
    #         assert (
    #             0 < x < self.width and 0 < y < self.height
    #         ), f"Coordinates {(x, y)} are out of bounds."

    #         # text
    #         if isinstance(entity, Agent):
    #             if entity.silent:
    #                 continue
    #             if np.all(entity.state.c == 0):
    #                 word = "_"
    #             elif self.continuous_actions:
    #                 word = (
    #                     "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
    #                 )
    #             else:
    #                 word = alphabet[np.argmax(entity.state.c)]

    #             message = entity.name + " sends " + word + "   "
    #             message_x_pos = self.width * 0.05
    #             message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
    #             self.game_font.render_to(
    #                 self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
    #             )
    #             text_line += 1

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = "polar"
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == "polar":
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == "grid":
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {"runtime.vectorized": True, "render.modes": ["human", "rgb_array"]}

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def _step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {"n": []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i : (i + env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def _reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def _render(self, mode="human", close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
