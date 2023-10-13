import gc
import multiprocessing
import os
import pickle
import sys
import time
from collections import deque
from copy import deepcopy
from multiprocessing import Pipe, connection
from multiprocessing.context import Process

import numpy as np
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

    def _generate_samples(self, obs, returns, use_history: bool, ctde: bool = False):
        Config = self.Config

        if ctde:
            samples = []
            for agent_idx in range(Config.n_agents):
                if use_history:
                    conditions = {
                        (0, Config.history_horizon + 1): to_torch(
                            obs[:, :, agent_idx], device=Config.device
                        ),
                        "agent_idx": to_torch(
                            [[agent_idx]], device=Config.device
                        ).repeat(obs.shape[0], 1, 1),
                    }
                else:
                    conditions = {
                        0: to_torch(obs[:, agent_idx], device=Config.device),
                        "agent_idx": to_torch([agent_idx], device=Config.device).repeat(
                            obs.shape[0], 1
                        ),
                    }

                samples_ = self.trainer.ema_model.conditional_sample(
                    conditions, returns=returns
                )
                samples.append(samples_[..., agent_idx, :])
            samples = torch.stack(samples, dim=-2)

        else:
            if use_history:
                conditions = {
                    (0, Config.history_horizon + 1): to_torch(obs, device=Config.device)
                }
            else:
                conditions = {0: to_torch(obs, device=Config.device)}
            samples = self.trainer.ema_model.conditional_sample(
                conditions,
                returns=returns,
                use_ddim_sample=getattr(Config, "use_ddim_sample", False),
            )

        if use_history:
            samples = samples[:, Config.history_horizon :]
        return samples

    def _evaluate(self, load_step=None):
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
        if Config.env_type == "smac":
            episode_wins = []

        cur_num_eval = 0
        while cur_num_eval < num_eval:
            num_episodes = min(num_eval - cur_num_eval, num_envs)
            rets = self._episodic_eval(num_episodes=num_episodes)
            episode_rewards.append(rets[1])
            if Config.env_type == "smac":
                episode_wins.append(rets[2])

            if cur_num_eval == 0:
                recorded_obs = rets[0]
                savepath = os.path.join("images", "sample-executed.png")
                self.trainer.renderer.composite(savepath, recorded_obs)

            cur_num_eval += num_episodes

        episode_rewards = np.concatenate(episode_rewards, axis=0)
        if Config.env_type == "smac":
            episode_wins = np.concatenate(episode_wins, axis=0)

        metrics_dict = dict(
            average_ep_reward=np.mean(episode_rewards, axis=0),
            std_ep_reward=np.std(episode_rewards, axis=0),
        )

        if Config.env_type == "smac":
            metrics_dict["win_rate"] = np.mean(episode_wins)

        logger.print(
            ", ".join([f"{k}: {v}" for k, v in metrics_dict.items()]),
            color="green",
        )
        logger.save_json(
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics_dict.items()
            },
            f"results/step_{load_step}-ep_{num_eval}-ddim.json"
            if getattr(Config, "use_ddim_sample", False)
            else f"results/step_{load_step}-ep_{num_eval}.json",
        )

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

        if Config.loader in [
            "datasets.CTDESequenceDataset",
            "datasets.CTDEHistoryCondSequenceDataset",
        ]:
            print("\n Using CTDE Evaluation \n")
            use_ctde = True
        else:
            assert (
                "CTDE" not in Config.loader
            ), f"Unknown CTDE dataset `{Config.loader}`"
            print("\n Using CTCE Evaluation \n")
            use_ctde = False

        dones = [0 for _ in range(num_episodes)]
        episode_rewards = [np.zeros(Config.n_agents) for _ in range(num_episodes)]
        if Config.env_type == "smac":
            episode_wins = np.zeros(num_episodes)

        returns = to_device(
            Config.test_ret * torch.ones(num_episodes, 1, Config.n_agents), device
        )

        t = 0
        obs_list = [env.reset()[None] for env in self.env_list[:num_episodes]]
        obs = np.concatenate(obs_list, axis=0)
        recorded_obs = [deepcopy(obs[:, None])]

        if hasattr(Config, "history_horizon") and Config.history_horizon > 0:
            print(f"\nUsing history length of {Config.history_horizon}\n")
            use_history = True
            obs_queue = deque(maxlen=Config.history_horizon + 1)
            obs_queue.extend(
                [np.zeros_like(obs) for _ in range(Config.history_horizon)]
            )
        else:
            use_history = False

        while sum(dones) < num_episodes:
            obs = self.normalizer.normalize(obs, "observations")
            if getattr(Config, "abs_pos", False):
                obs[..., 4 : -(Config.n_agents - 1) * 2] = obs[
                    ..., 4 : -(Config.n_agents - 1) * 2
                ] + np.tile(obs[..., 2:4], 5)

            if use_history:
                obs_queue.append(obs)
                obs = np.stack(list(obs_queue), axis=1)

            samples = self._generate_samples(obs, returns, use_history, ctde=use_ctde)

            obs_comb = torch.cat([samples[:, 0, :, :], samples[:, 1, :, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, Config.n_agents, 2 * observation_dim)
            if getattr(Config, "ego_only_inv", False):
                ego_observation_dim = 4
                obs_comb = torch.cat(
                    [
                        obs_comb[..., :ego_observation_dim],
                        obs_comb[
                            ..., observation_dim : observation_dim + ego_observation_dim
                        ],
                    ],
                    dim=-1,
                )

            if Config.share_inv:
                action = self.trainer.ema_model.inv_model(obs_comb)
            else:
                action = torch.stack(
                    [
                        self.trainer.ema_model.inv_model[i](obs_comb[:, i])
                        for i in range(Config.n_agents)
                    ],
                    dim=1,
                )

            samples = to_np(samples)
            action = to_np(action)

            if self.discrete_action:
                legal_action = np.stack(
                    [env.get_legal_actions() for env in self.env_list], axis=0
                )
                action[np.where(legal_action.astype(int) == 0)] = -np.inf
                action = np.argmax(action, axis=-1)
            else:
                action = self.normalizer.unnormalize(action, "actions")

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
                    if use_history:
                        obs_list.append(obs[i, 0][None])
                    else:
                        obs_list.append(obs[i][None])

                else:
                    this_obs, this_reward, this_done, this_info = self.env_list[i].step(
                        action[i]
                    )
                    obs_list.append(this_obs[None])

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

        recorded_obs = np.concatenate(recorded_obs, axis=1)
        episode_rewards = np.array(episode_rewards)

        if Config.env_type == "smac":
            return recorded_obs, episode_rewards, episode_wins
        else:
            return recorded_obs, episode_rewards

    def _init(self, log_dir, **kwargs):
        assert self.initialized is False, "Evaluator can only be initialized once."

        self.log_dir = log_dir
        with open(os.path.join(log_dir, "parameters.pkl"), "rb") as f:
            params = pickle.load(f)

        Config = build_config_from_dict(params["Config"])
        self.Config = Config = build_config_from_dict(kwargs, Config)
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

        dataset = dataset_config()
        self.normalizer = dataset.normalizer
        del dataset
        gc.collect()

        renderer = render_config()
        model = model_config()
        diffusion = diffusion_config(model)
        self.trainer = trainer_config(diffusion, None, renderer)

        self.discrete_action = False

        """ Load Environment """
        if Config.env_type == "d4rl":
            from diffuser.datasets.d4rl import load_environment
        elif Config.env_type == "ma_mujoco":
            from diffuser.datasets.ma_mujoco import load_environment
        elif Config.env_type == "mpe":
            from diffuser.datasets.mpe import load_environment
        elif Config.env_type == "smac":
            from diffuser.datasets.smac import load_environment

            self.discrete_action = True
        else:
            raise NotImplementedError(f"{Config.env_type} not implemented")

        Config.num_envs = getattr(Config, "num_envs", Config.num_eval)
        self.env_list = [
            load_environment(Config.dataset) for _ in range(Config.num_envs)
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
