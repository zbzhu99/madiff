import os
import pickle

import numpy as np
import torch
from ml_logger import logger

import diffuser.utils as utils
from diffuser.utils.arrays import batch_to_device, to_np
from diffuser.utils.launcher_util import build_config_from_dict
from diffuser.utils.training import cycle


class MADOfflineEvaluator:
    def init(
        self,
        log_dir: str,
        batch_size: int,
        num_batches: int,
        **kwargs,
    ):
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

        self.dataset = dataset_config()
        self.normalizer = self.dataset.normalizer
        self.dataloader = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
        )

        renderer = render_config()
        model = model_config()
        diffusion = diffusion_config(model)
        self.trainer = trainer_config(diffusion, None, renderer)

        self.discrete_action = False

        """ Load Environment """
        if Config.env_type == "smac":
            self.discrete_action = True
        else:
            self.discrete_action = False

        self.initialized = True

    def evaluate(self, load_step: int):
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

        batch = next(self.dataloader)
        batch = batch_to_device(batch, device=self.Config.device)

        conditions = batch.cond
        if self.trainer.ema_model.returns_condition:
            returns = batch.returns
        else:
            returns = None

        plan_obs = self.trainer.ema_model.conditional_sample(
            conditions, returns=returns
        )

        plan_obs_comb = torch.cat([plan_obs[:, :-1], plan_obs[:, 1:]], dim=-1)
        plan_acts = self.trainer.ema_model.inv_model(plan_obs_comb)

        obs = batch.x[..., self.dataset.action_dim :]
        acts = batch.x[..., : self.dataset.action_dim]

        obs_comb = torch.cat([obs[:, :-1], obs[:, 1:]], dim=-1)
        pred_acts = self.trainer.ema_model.inv_model(obs_comb)

        obs = to_np(obs)
        acts = to_np(acts)
        plan_obs = to_np(plan_obs)
        plan_acts = to_np(plan_acts)
        pred_acts = to_np(pred_acts)

        plan_obs_mse = np.mean((plan_obs - obs) ** 2)
        plan_act_mse = np.mean((plan_acts - acts[:, :-1]) ** 2)
        pred_act_mse = np.mean((pred_acts - acts[:, :-1]) ** 2)

        plan_obs_mse_first_step = np.mean((plan_obs[:, 1] - obs[:, 1]) ** 2)
        plan_act_mse_first_step = np.mean((plan_acts[:, 0] - acts[:, 0]) ** 2)
        pred_act_mse_first_step = np.mean((pred_acts[:, 0] - acts[:, 0]) ** 2)

        metrics_dict = dict(
            plan_obs_mse=plan_obs_mse.item(),
            plan_act_mse=plan_act_mse.item(),
            pred_act_mse=pred_act_mse.item(),
            plan_obs_mse_first_step=plan_obs_mse_first_step.item(),
            plan_act_mse_first_step=plan_act_mse_first_step.item(),
            pred_act_mse_first_step=pred_act_mse_first_step.item(),
        )

        save_file_path = os.path.join(
            f"results/step_{load_step}-ddim-offline.json"
            if getattr(Config, "use_ddim_sample", False)
            else f"results/step_{load_step}-offline.json",
        )
        logger.save_json(
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics_dict.items()
            },
            save_file_path,
        )
