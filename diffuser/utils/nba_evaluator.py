import gc
import os
import pickle
import numpy as np
import torch
from ml_logger import logger

import diffuser.utils as utils
from diffuser.utils.launcher_util import build_config_from_dict
from diffuser.datasets.sequence import NBASequenceDataset, NBAHistoryCondSequenceDataset
from .arrays import batch_to_device


class NBAEvaluator:
    def __init__(self, **kwargs):
        # dummy args
        self.args = {}
        for key in kwargs:
            self.args[key] = kwargs[key]

    def init(self, **kwargs):
        self.log_dir = kwargs["log_dir"]
        self.batch_idx = kwargs["batch_idx"]
        self.batch_size = kwargs["batch_size"]
        print(f"\n\nEvaluating {self.batch_idx}-th batch of batch size {self.batch_size}\n\n")
        with open(os.path.join(self.log_dir, "parameters.pkl"), "rb") as f:
            params = pickle.load(f)

        logger.configure(self.log_dir)
        Config = build_config_from_dict(params["Config"])
        self.Config = Config = build_config_from_dict(kwargs, Config)
        self.Config.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        with open(os.path.join(self.log_dir, "model_config.pkl"), "rb") as f:
            model_config = pickle.load(f)

        with open(os.path.join(self.log_dir, "diffusion_config.pkl"), "rb") as f:
            diffusion_config = pickle.load(f)

        with open(os.path.join(self.log_dir, "dataset_config.pkl"), "rb") as f:
            dataset_config = pickle.load(f)

        dataset = dataset_config()
        self.normalizer = dataset.normalizer
        assert dataset.env.metadata["HZ"] == kwargs["nba_hz"]
        del dataset
        gc.collect()

        model = model_config()
        self.diffusion = diffusion_config(model)

        if kwargs["use_history"] == False:
            dataset = NBASequenceDataset(
                env_type="nba",
                env="test",
                n_agents=10,
                horizon=kwargs["horizon"],
                use_action=False,
                max_path_length=20000,
                nba_eval_valid_samples=1000,
            )
            self.history_horizon = 0
        else:
            dataset = NBAHistoryCondSequenceDataset(
                env_type="nba",
                env="test",
                n_agents=10,
                horizon=kwargs["horizon"],
                use_action=False,
                max_path_length=20000,
                nba_eval_valid_samples=1000,
                nba_hz=kwargs["nba_hz"],
                history_horizon=kwargs["history_horizon"],
            )
            self.history_horizon = kwargs["history_horizon"]

        dataset.normalizer = self.normalizer
        dataset.normalize()
        print("Finish normalizing with training normalizer")

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=kwargs["batch_size"],
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def displacement_error(self, pred, truth, history_horizon=0, norm=2, mode="mean"):
        assert (mode == "mean" or mode == "final") and (norm == 2 or norm == 1)
        pred, truth = pred[:, history_horizon:, :, :], truth[:, history_horizon:, :, :]
        (pred, truth) = (
            (pred[:, -1, :, :].unsqueeze(1), truth[:, -1, :, :].unsqueeze(1))
            if mode == "final"
            else (pred, truth)
        )
        return torch.mean(torch.norm(pred - truth, p=norm, dim=3)).item()

    def min_displacement_error(
        self, pred, truth, history_horizon=0, norm=2, mode="mean"
    ):
        assert len(pred.shape) == 5
        assert (mode == "mean" or mode == "final") and (norm == 2 or norm == 1)
        pred, truth = (
            pred[:, :, history_horizon:, :, :],
            truth[:, history_horizon:, :, :],
        )
        (pred, truth) = (
            (pred[:, :, -1, :, :].unsqueeze(2), truth[:, -1, :, :].unsqueeze(1))
            if mode == "final"
            else (pred, truth)
        )
        return torch.mean(
            torch.min(
                torch.mean(torch.norm(pred - truth, p=norm, dim=4), dim=2), dim=0
            ).values
        ).item()

    def evaluate(self, **kwargs):
        Config = self.Config
        load_step = kwargs["load_step"]
        loadpath = os.path.join(self.log_dir, "checkpoint")

        utils.set_seed(Config.seed)

        if Config.save_checkpoints:
            assert load_step is not None
            loadpath = os.path.join(loadpath, f"state_{load_step}.pt")
        else:
            loadpath = os.path.join(loadpath, "state.pt")

        state_dict = torch.load(loadpath, map_location=Config.device)
        state_dict["ema"] = {
            k: v
            for k, v in state_dict["ema"].items()
            if "value_diffusion_model." not in k
        }

        self.diffusion.load_state_dict(state_dict["ema"])

        sample_times = kwargs["sample_times"]
        assert sample_times == 1
        min_batch_loss = []
        mean_batch_loss = []
        ADE = []
        FDE = []
        min_ADE = []
        min_FDE = []
        for idx, batch in enumerate(self.dataloader):
            if idx != self.batch_idx:
                continue
            batch = batch_to_device(batch, device=Config.device)
            batch_condition = batch[1]
            obs_truth = batch[0]
            unnormed_obs_truth = torch.tensor(
                self.normalizer.unnormalize(obs_truth.cpu(), "observations")
            )
            pres_batch_loss_list = []
            unnormed_obs_sample_list = []
            for _ in range(20):
                unnormed_obs_sample_distribution = []
                for _ in range(sample_times):
                    sample = self.diffusion.conditional_sample(batch_condition)
                    obs_sample = (
                        sample[:, :, :, 1:-1] if sample.shape[-1] == 4 else sample
                    )  # bz, t, a, f
                    unnormed_obs_sample = torch.tensor(
                        self.normalizer.unnormalize(
                            obs_sample.detach().cpu(), "observations"
                        )
                    )
                    pres_batch_loss_list.append(
                        torch.nn.functional.mse_loss(
                            unnormed_obs_sample, unnormed_obs_truth
                        ).item()
                    )
                    unnormed_obs_sample_distribution.append(unnormed_obs_sample)
                unnormed_obs_sample_list.append(
                    torch.mean(
                        torch.stack(unnormed_obs_sample_distribution, dim=0), dim=0
                    )
                )
            if len(unnormed_obs_sample_distribution) > 1:
                unnormed_obs_sample_list.append(torch.mean(torch.stack(unnormed_obs_sample_distribution, dim=0), dim=0))
            else:
                unnormed_obs_sample_list.append(unnormed_obs_sample_distribution[0])

            stacked_unnormed_obs_sample = torch.stack(unnormed_obs_sample_list, dim=0)
            min_ADE.append(
                self.min_displacement_error(
                    stacked_unnormed_obs_sample,
                    unnormed_obs_truth,
                    history_horizon=self.history_horizon,
                    norm=2,
                    mode="mean",
                )
            )
            min_FDE.append(
                self.min_displacement_error(
                    stacked_unnormed_obs_sample,
                    unnormed_obs_truth,
                    history_horizon=self.history_horizon,
                    norm=2,
                    mode="final",
                )
            )
            mean_unnormed_obs_sample = torch.mean(
                torch.stack(unnormed_obs_sample_list, dim=0), dim=0
            )
            mean_batch_loss.append(
                torch.nn.functional.mse_loss(
                    mean_unnormed_obs_sample, unnormed_obs_truth
                ).item()
            )
            ADE.append(
                self.displacement_error(
                    mean_unnormed_obs_sample,
                    unnormed_obs_truth,
                    history_horizon=self.history_horizon,
                    norm=2,
                    mode="mean",
                )
            )
            FDE.append(
                self.displacement_error(
                    mean_unnormed_obs_sample,
                    unnormed_obs_truth,
                    history_horizon=self.history_horizon,
                    norm=2,
                    mode="final",
                )
            )

            print("present min batch mse loss is: ", min(pres_batch_loss_list))
            print("present mean batch mse loss is: ", mean_batch_loss[-1])
            print("present mean batch ADE is: ", ADE[-1])
            print("present mean batch FDE is: ", FDE[-1])
            print("present batch min ADE is: ", min_ADE[-1])
            print("present batch min FDE is: ", min_FDE[-1])
            min_batch_loss.append(min(pres_batch_loss_list))

        min_pred_loss_mean = sum(min_batch_loss) / len(min_batch_loss)
        min_pred_loss_std = np.std(min_batch_loss)
        mean_pred_loss_mean = sum(mean_batch_loss) / len(mean_batch_loss)
        mean_pred_loss_std = np.std(mean_batch_loss)
        mean_ADE = sum(ADE) / len(ADE)
        mean_FDE = sum(FDE) / len(FDE)
        min_ADE = sum(min_ADE) / len(min_ADE)
        min_FDE = sum(min_FDE) / len(min_FDE)

        logger.print("MSE loss is: ", mean_pred_loss_mean)
        logger.print("ADE is: ", mean_ADE)
        logger.print("FDE is: ", mean_FDE)
        logger.print("min ADE is: ", min_ADE)
        logger.print("min FDE is: ", min_FDE)
        logger.print("min ADE is: ", min_ADE)
        logger.print("min FDE is: ", min_FDE)
        logger.save_json(
            {
                "mean_pred_loss_mean": mean_pred_loss_mean,
                "mean_pred_loss_std": mean_pred_loss_std,
                "min_pred_loss_mean": min_pred_loss_mean,
                "min_pred_loss_std": min_pred_loss_std,
                "ADE": mean_ADE,
                "FDE": mean_FDE,
                "min_ADE": min_ADE,
                "min_FDE": min_FDE,
            },
            f"results/bid_{self.batch_idx}_{self.batch_size}_step{load_step}.json",
        )

        np_traj = np.array(torch.stack(unnormed_obs_sample_list, dim=0))
        logger.save_pkl(
            np_traj,
            f"results/obs_samples_bid_{self.batch_idx}_{self.batch_size}_step{load_step}.pkl",
        )
