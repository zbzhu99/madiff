from typing import Optional
import gc
import multiprocessing
import os
import pickle
import sys
import time
import importlib
from collections import deque
from copy import deepcopy, copy
from multiprocessing import Pipe, connection
from multiprocessing.context import Process

import numpy as np
import einops
import torch
from ml_logger import logger

import diffuser.utils as utils
from diffuser.utils.arrays import to_device, to_np, to_torch
from diffuser.utils.launcher_util import build_config_from_dict


class MADEvaluatorWorker(Process):
    def __init__(
        self,
        parent_remote: connection.Connection,
        child_remote: connection.Connection,
        queue: multiprocessing.Queue,
        verbose: bool = False,
    ):
        self.parent_remote = parent_remote
        self.p = child_remote
        self.queue = queue
        self.initialized = False
        self.verbose = verbose
        super().__init__()

    def _generate_samples(self, obs, returns, env_ts):
        Config = self.Config

        env_ts = env_ts.clone()
        env_ts[torch.where(env_ts < 0)] = Config.max_path_length
        env_ts[torch.where(env_ts >= Config.max_path_length)] = Config.max_path_length

        attention_masks = np.zeros(
            (obs.shape[0], Config.horizon + Config.history_horizon, Config.n_agents, 1)
        )
        attention_masks[:, Config.history_horizon :] = 1.0

        shape = (
            obs.shape[0],
            Config.horizon + Config.history_horizon,
            *obs.shape[-2:],
        )  # b t a f
        if Config.decentralized_execution:
            joint_cond_trajectories, joint_cond_masks, joint_attention_masks = (
                [],
                [],
                [],
            )
            for a_idx in range(Config.n_agents):
                local_cond_trajectories = np.zeros(shape, dtype=obs.dtype)
                local_cond_trajectories[:, : Config.history_horizon + 1, a_idx] = obs[
                    :, :, a_idx
                ]

                agent_mask = np.zeros(Config.n_agents)
                agent_mask[a_idx] = 1.0
                local_cond_masks = self.mask_generator(shape, agent_mask)

                local_attention_masks = copy(attention_masks)
                local_attention_masks[:, : Config.history_horizon, a_idx] = 1.0

                joint_cond_trajectories.append(
                    to_torch(local_cond_trajectories, device=Config.device)
                )
                joint_cond_masks.append(
                    to_torch(local_cond_masks, device=Config.device)
                )
                joint_attention_masks.append(
                    to_torch(local_attention_masks, device=Config.device)
                )

            joint_cond_trajectories = einops.rearrange(
                torch.stack(joint_cond_trajectories, dim=1), "b a ... -> (b a) ..."
            )
            joint_cond_masks = einops.rearrange(
                torch.stack(joint_cond_masks, dim=1), "b a ... -> (b a) ..."
            )
            joint_attention_masks = einops.rearrange(
                torch.stack(joint_attention_masks, dim=1), "b a ... -> (b a) ..."
            )
            conditions = {
                "x": joint_cond_trajectories,
                "masks": joint_cond_masks,
            }
            returns = einops.repeat(returns, "b ... -> (b a) ...", a=Config.n_agents)
            env_ts = einops.repeat(env_ts, "b ... -> (b a) ...", a=Config.n_agents)

            joint_samples = self.trainer.ema_model.conditional_sample(
                conditions,
                returns=returns,
                env_ts=env_ts,
                attention_masks=joint_attention_masks,
            )
            joint_samples = einops.rearrange(
                joint_samples, "(b a) ... -> b a ...", a=Config.n_agents
            )

            samples = []
            for a_idx in range(Config.n_agents):
                samples.append(joint_samples[:, a_idx, ..., a_idx, :])
            samples = torch.stack(samples, dim=-2)

        else:
            cond_trajectories = np.zeros(shape, dtype=obs.dtype)
            cond_trajectories[:, : Config.history_horizon + 1] = obs
            agent_mask = np.ones(Config.n_agents)
            cond_masks = self.mask_generator(shape, agent_mask)
            conditions = {
                "x": to_torch(cond_trajectories, device=Config.device),
                "masks": to_torch(cond_masks, device=Config.device),
            }
            attention_masks[:, : Config.history_horizon] = 1.0
            attention_masks = to_torch(attention_masks, device=Config.device)
            samples = self.trainer.ema_model.conditional_sample(
                conditions,
                returns=returns,
                env_ts=env_ts,
                attention_masks=attention_masks,
            )

        samples = samples[:, Config.history_horizon :]
        return samples

    def _evaluate(self, load_step: Optional[int] = None):
        assert (
            self.initialized is True
        ), "Evaluator should be initialized before evaluation."

        Config = self.Config
        loadpath = os.path.join(self.log_dir, "checkpoint")

        utils.set_seed(Config.seed)

        if Config.save_checkpoints:
            assert load_step is not None
            loadpath = os.path.join(loadpath, f"state_{load_step}.pt")
        else:
            loadpath = os.path.join(loadpath, "state.pt")

        state_dict = torch.load(loadpath, map_location=Config.device)
        state_dict["model"] = {
            k: v
            for k, v in state_dict["model"].items()
            if "value_diffusion_model." not in k
        }
        state_dict["ema"] = {
            k: v
            for k, v in state_dict["ema"].items()
            if "value_diffusion_model." not in k
        }

        self.trainer.step = state_dict["step"]
        self.trainer.model.load_state_dict(state_dict["model"])
        self.trainer.ema_model.load_state_dict(state_dict["ema"])

        num_eval = Config.num_eval
        num_envs = Config.num_envs

        episode_rewards = []
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            episode_wins = []

        cur_num_eval = 0
        while cur_num_eval < num_eval:
            num_episodes = min(num_eval - cur_num_eval, num_envs)
            rets = self._episodic_eval(num_episodes=num_episodes)
            episode_rewards.append(rets[1])
            if Config.env_type == "smac" or Config.env_type == "smacv2":
                episode_wins.append(rets[2])

            cur_num_eval += num_episodes

        episode_rewards = np.concatenate(episode_rewards, axis=0)
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            episode_wins = np.concatenate(episode_wins, axis=0)

        metrics_dict = dict(
            average_ep_reward=np.mean(episode_rewards, axis=0),
            std_ep_reward=np.std(episode_rewards, axis=0),
        )

        if Config.env_type == "smac" or Config.env_type == "smacv2":
            metrics_dict["win_rate"] = np.mean(episode_wins)

        logger.print(
            ", ".join([f"{k}: {v}" for k, v in metrics_dict.items()]),
            color="green",
        )
        save_file_path = (
            f"results/step_{load_step}-ep_{num_eval}-ddim.json"
            if getattr(Config, "use_ddim_sample", False)
            else f"results/step_{load_step}-ep_{num_eval}.json"
        )
        if self.rewrite_cgw:
            save_file_path = save_file_path.replace(
                ".json", f"-cg_{self.trainer.ema_model.condition_guidance_w}.json"
            )
        logger.save_json(
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics_dict.items()
            },
            save_file_path,
        )

    def _update_return_to_go(self, rtg, reward):
        rtg = rtg * self.Config.returns_scale
        reward = torch.tensor(reward, device=rtg.device, dtype=rtg.dtype).reshape(1, -1)
        rtg = (rtg - reward) / self.Config.discount
        rtg = rtg / self.Config.returns_scale
        return rtg

    def _episodic_eval(self, num_episodes: int):
        """Evaluate for one episode each environment."""

        # `num_episodes` can be smaller than total number of environment, and
        # we only use the first `num_episodes` environments.
        assert (
            num_episodes <= self.Config.num_envs
        ), f"num_episodes should be <= num_envs, but {num_episodes} > {self.Config.num_envs}"

        Config = self.Config
        device = Config.device
        observation_dim = self.normalizer.observation_dim

        dones = [0 for _ in range(num_episodes)]
        episode_rewards = [np.zeros(Config.n_agents) for _ in range(num_episodes)]
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            episode_wins = np.zeros(num_episodes)

        returns = to_device(
            Config.test_ret * torch.ones(num_episodes, 1, Config.n_agents), device
        )
        env_ts = to_device(
            torch.arange(Config.horizon + Config.history_horizon)
            - Config.history_horizon,
            device,
        )
        env_ts = einops.repeat(env_ts, "t -> b t", b=num_episodes)

        t = 0
        obs_list = [env.reset()[None] for env in self.env_list[:num_episodes]]
        obs = np.concatenate(obs_list, axis=0)
        recorded_obs = [deepcopy(obs[:, None])]

        if Config.history_horizon > 0:
            print(f"\nUsing history length of {Config.history_horizon}\n")
        else:
            print("\nDo NOT use history conditioning\n")
        obs_queue = deque(maxlen=Config.history_horizon + 1)
        if Config.use_zero_padding:
            obs_queue.extend(
                [np.zeros_like(obs) for _ in range(Config.history_horizon)]
            )
        else:
            normed_obs = self.normalizer.normalize(obs, "observations")
            obs_queue.extend([normed_obs for _ in range(Config.history_horizon)])

        while sum(dones) < num_episodes:
            obs = self.normalizer.normalize(obs, "observations")
            obs_queue.append(obs)
            obs = np.stack(list(obs_queue), axis=1)

            samples = self._generate_samples(obs, returns, env_ts)

            obs_comb = torch.cat([samples[:, 0, :, :], samples[:, 1, :, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, Config.n_agents, 2 * observation_dim)

            if Config.share_inv or Config.joint_inv:
                if Config.joint_inv:
                    actions = self.trainer.ema_model.inv_model(
                        obs_comb.reshape(obs_comb.shape[0], -1)
                    ).reshape(obs_comb.shape[0], obs_comb.shape[1], -1)
                else:
                    actions = self.trainer.ema_model.inv_model(obs_comb)
            else:
                actions = torch.stack(
                    [
                        self.trainer.ema_model.inv_model[i](obs_comb[:, i])
                        for i in range(Config.n_agents)
                    ],
                    dim=1,
                )

            samples = to_np(samples)
            actions = to_np(actions)

            if self.discrete_action:
                legal_action = np.stack(
                    [env.get_legal_actions() for env in self.env_list], axis=0
                )
                actions[np.where(legal_action.astype(int) == 0)] = -np.inf
                actions = np.argmax(actions, axis=-1)
            else:
                actions = self.normalizer.unnormalize(actions, "actions")

            if t == 0:
                normed_observations = samples[:, :, :, :]
                observations = self.normalizer.unnormalize(
                    normed_observations, "observations"
                )
                savepath = os.path.join("images", "sample-planned.png")
                self.trainer.renderer.composite(savepath, observations)

            obs_list = []
            for i in range(num_episodes):
                if dones[i] == 1:
                    obs_list.append(obs[i, 0][None])
                else:
                    this_obs, this_reward, this_done, this_info = self.env_list[i].step(
                        actions[i]
                    )
                    obs_list.append(this_obs[None])

                    if Config.use_return_to_go:
                        returns[i] = self._update_return_to_go(returns[i], this_reward)

                    if this_done.all() or t >= Config.max_path_length - 1:
                        dones[i] = 1
                        episode_rewards[i] += this_reward
                        if "battle_won" in this_info.keys():
                            episode_wins[i] = this_info["battle_won"]
                            logger.print(
                                f"Episode ({i}): battle won {episode_wins[i]}",
                                color="green",
                            )

                        logger.print(
                            f"Episode ({i}): {episode_rewards[i]}", color="green"
                        )

                    else:
                        episode_rewards[i] += this_reward

            obs = np.concatenate(obs_list, axis=0)
            recorded_obs.append(deepcopy(obs[:, None]))
            t += 1
            env_ts = env_ts + 1

        recorded_obs = np.concatenate(recorded_obs, axis=1)
        episode_rewards = np.array(episode_rewards)

        if Config.env_type == "smac" or Config.env_type == "smacv2":
            return recorded_obs, episode_rewards, episode_wins
        else:
            return recorded_obs, episode_rewards

    def _init(
        self, log_dir: str, condition_guidance_w: Optional[float] = None, **kwargs
    ):
        assert self.initialized is False, "Evaluator can only be initialized once."

        self.log_dir = log_dir
        with open(os.path.join(log_dir, "parameters.pkl"), "rb") as f:
            params = pickle.load(f)

        Config = build_config_from_dict(params["Config"])
        self.Config = Config = build_config_from_dict(kwargs, Config)
        self.Config.joint_inv = getattr(Config, "joint_inv", False)
        self.Config.use_return_to_go = getattr(Config, "use_return_to_go", False)
        self.Config.use_ddim_sample = getattr(Config, "use_ddim_sample", False)
        self.Config.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.configure(log_dir)
        torch.backends.cudnn.benchmark = True

        with open(os.path.join(log_dir, "model_config.pkl"), "rb") as f:
            model_config = pickle.load(f)

        with open(os.path.join(log_dir, "diffusion_config.pkl"), "rb") as f:
            diffusion_config = pickle.load(f)

        with open(os.path.join(log_dir, "trainer_config.pkl"), "rb") as f:
            trainer_config = pickle.load(f)

        with open(os.path.join(log_dir, "dataset_config.pkl"), "rb") as f:
            dataset_config = pickle.load(f)

        with open(os.path.join(log_dir, "render_config.pkl"), "rb") as f:
            render_config = pickle.load(f)

        self.rewrite_cgw = False
        if condition_guidance_w is not None:
            print(f"Set condition guidance weight to {condition_guidance_w}")
            diffusion_config._dict["condition_guidance_w"] = condition_guidance_w
            self.rewrite_cgw = True

        dataset = dataset_config()
        self.normalizer = dataset.normalizer
        self.mask_generator = dataset.mask_generator
        del dataset
        gc.collect()

        renderer = render_config()
        model = model_config()
        diffusion = diffusion_config(model)
        self.trainer = trainer_config(diffusion, None, renderer)

        if Config.use_ddim_sample:
            print(f"\n Use DDIM Sampler of {Config.n_ddim_steps} Step(s) \n")
            self.trainer.model.set_ddim_scheduler(Config.n_ddim_steps)
            self.trainer.ema_model.set_ddim_scheduler(Config.n_ddim_steps)

        self.discrete_action = False
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            self.discrete_action = True

        """ Load Environment """
        env_mod_name = {
            "d4rl": "diffuser.datasets.d4rl",
            "mahalfcheetah": "diffuser.datasets.mahalfcheetah",
            "mamujoco": "diffuser.datasets.mamujoco",
            "mpe": "diffuser.datasets.mpe",
            "smac": "diffuser.datasets.smac_env",
            "smacv2": "diffuser.datasets.smacv2_env",
        }[Config.env_type]
        env_mod = importlib.import_module(env_mod_name)

        Config.num_envs = getattr(Config, "num_envs", Config.num_eval)
        self.env_list = [
            env_mod.load_environment(Config.dataset) for _ in range(Config.num_envs)
        ]
        self.initialized = True

    def run(self):
        self.parent_remote.close()
        if not self.verbose:
            sys.stdout = open(os.devnull, "w")
        try:
            while True:
                try:
                    cmd, data = self.queue.get()
                except EOFError:
                    self.p.close()
                    break

                if cmd == "init":
                    self._init(**data)
                elif cmd == "evaluate":
                    self._evaluate(**data)
                elif cmd == "close":
                    self.p.send("closed")
                    self.p.close()
                    # self.queue.shutdown()
                    break
                else:
                    self.p.close()
                    raise NotImplementedError(f"Unknown command {cmd}")

                time.sleep(1)

        except KeyboardInterrupt:
            self.p.close()


class MADEvaluator:
    def __init__(self, **kwargs):
        multiprocessing.set_start_method("spawn", force=True)
        self.parent_remote, self.child_remote = Pipe()
        self.queue = multiprocessing.Queue()
        self._worker_process = MADEvaluatorWorker(
            parent_remote=self.parent_remote,
            child_remote=self.child_remote,
            queue=self.queue,
            **kwargs,
        )
        self._worker_process.start()
        self.child_remote.close()

    def init(self, **kwargs):
        self.queue.put(["init", kwargs])

    def evaluate(self, **kwargs):
        self.queue.put(["evaluate", kwargs])

    def __del__(self):
        try:
            self.queue.put(["close", None])
            # mp may be deleted so it may raise AttributeError
            self.parent_remote.recv()
            self._worker_process.join()
        except (BrokenPipeError, EOFError, AttributeError, FileNotFoundError):
            pass
        # ensure the subproc is terminated
        self._worker_process.terminate()
