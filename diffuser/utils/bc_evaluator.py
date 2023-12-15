import gc
import multiprocessing
import os
import pickle
import sys
import time
from copy import deepcopy
from multiprocessing import Pipe, connection
from multiprocessing.context import Process

import numpy as np
import torch
from ml_logger import logger

import diffuser.utils as utils
from diffuser.utils.arrays import to_np, to_torch
from diffuser.utils.launcher_util import build_config_from_dict


class BCEvaluatorWorker(Process):
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
        self.trainer.step = state_dict["step"]
        self.trainer.model.load_state_dict(state_dict["model"])

        num_eval = Config.num_eval
        device = Config.device

        dones = [0 for _ in range(num_eval)]
        episode_rewards = [np.zeros(Config.n_agents) for _ in range(num_eval)]

        t = 0
        obs_list = [env.reset()[None] for env in self.env_list]
        obs = np.concatenate(obs_list, axis=0)
        recorded_obs = [deepcopy(obs[:, None])]

        while sum(dones) < Config.num_eval:
            obs = self.normalizer.normalize(obs, "observations")
            obs = to_torch(
                self.normalizer.normalize(obs, "observations"), device=device
            )
            action = self.trainer.model(obs)
            action = to_np(action)
            action = self.normalizer.unnormalize(action, "actions")

            obs_list = []
            for i in range(num_eval):
                this_obs, this_reward, this_done, _ = self.env_list[i].step(action[i])
                obs_list.append(this_obs[None])
                if this_done.all() or t >= Config.max_path_length - 1:
                    if dones[i] == 1:
                        pass
                    else:
                        dones[i] = 1
                        episode_rewards[i] += this_reward
                        logger.print(
                            f"Episode ({i}): {episode_rewards[i]}", color="green"
                        )
                else:
                    if dones[i] == 1:
                        pass
                    else:
                        episode_rewards[i] += this_reward

            obs = np.concatenate(obs_list, axis=0)
            recorded_obs.append(deepcopy(obs[:, None]))
            t += 1

        recorded_obs = np.concatenate(recorded_obs, axis=1)
        episode_rewards = np.array(episode_rewards)

        logger.print(
            f"average_ep_reward: {np.mean(episode_rewards, axis=0)}, std_ep_reward: {np.std(episode_rewards, axis=0)}",
            color="green",
        )
        logger.save_json(
            {
                "average_ep_reward": np.mean(episode_rewards, axis=0).tolist(),
                "std_ep_reward": np.std(episode_rewards, axis=0).tolist(),
            },
            f"results/step_{load_step}.json",
        )

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
        # utils.set_seed(self.Config.seed)

        with open(os.path.join(log_dir, "model_config.pkl"), "rb") as f:
            model_config = pickle.load(f)

        with open(os.path.join(log_dir, "bc_config.pkl"), "rb") as f:
            bc_config = pickle.load(f)

        with open(os.path.join(log_dir, "trainer_config.pkl"), "rb") as f:
            trainer_config = pickle.load(f)

        with open(os.path.join(log_dir, "dataset_config.pkl"), "rb") as f:
            dataset_config = pickle.load(f)

        dataset = dataset_config()
        self.normalizer = dataset.normalizer
        del dataset
        gc.collect()

        model = model_config()
        bc = bc_config(model)
        self.trainer = trainer_config(bc, None)

        """ Load Environment """
        if Config.env_type == "d4rl":
            from diffuser.datasets.d4rl import load_environment
        elif Config.env_type == "ma_mujoco":
            from diffuser.datasets.ma_mujoco import load_environment
        elif Config.env_type == "mpe":
            from diffuser.datasets.mpe import load_environment
        else:
            raise NotImplementedError(f"{Config.env_type} not implemented")

        self.env_list = [
            load_environment(Config.dataset) for _ in range(Config.num_eval)
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


class BCEvaluator:
    def __init__(self, **kwargs):
        multiprocessing.set_start_method("spawn", force=True)
        self.parent_remote, self.child_remote = Pipe()
        self.queue = multiprocessing.Queue()
        self._worker_process = BCEvaluatorWorker(
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
