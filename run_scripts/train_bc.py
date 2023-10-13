import argparse
import os

import torch
import yaml

import diffuser.utils as utils
from diffuser.utils.launcher_util import (
    build_config_from_dict,
    discover_latest_checkpoint_path,
)


def main(Config, RUN):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    utils.set_seed(Config.seed)
    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    dataset_config = utils.Config(
        Config.loader,
        savepath="dataset_config.pkl",
        env_type=Config.env_type,
        env=Config.dataset,
        n_agents=Config.n_agents,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        max_n_episodes=Config.max_n_episodes,
        max_path_length=Config.max_path_length,
        agent_share_parameters=utils.config.import_class(
            Config.model
        ).agent_share_parameters,
    )

    dataset = dataset_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    model_config = utils.Config(
        Config.model,
        savepath="model_config.pkl",
        n_agents=Config.n_agents,
        observation_dim=observation_dim,
        action_dim=action_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        device=Config.device,
    )

    bc_config = utils.Config(
        Config.bc,
        savepath="bc_config.pkl",
        observation_dim=observation_dim,
        action_dim=action_dim,
        device=Config.device,
    )

    trainer_config = utils.Config(
        utils.BCTrainer,
        savepath="trainer_config.pkl",
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        save_freq=Config.save_freq,
        eval_freq=Config.eval_freq,
        log_freq=Config.log_freq,
        bucket=logger.root,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
    )

    evaluator_config = utils.Config(
        Config.evaluator,
        savepath="evaluator_config.pkl",
        verbose=False,
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()
    bc = bc_config(model)
    trainer = trainer_config(bc, dataset)

    if Config.eval_freq > 0:
        evaluator = evaluator_config()
        evaluator.init(log_dir=logger.prefix)
        trainer.set_evaluator(evaluator)

    if Config.continue_training:
        loadpath = discover_latest_checkpoint_path(
            os.path.join(trainer.bucket, logger.prefix, "checkpoint")
        )
        if loadpath is not None:
            state_dict = torch.load(loadpath, map_location=Config.device)
            logger.print(
                f"\nLoaded checkpoint from {loadpath} (step {state_dict['step']})\n",
                color="green",
            )
            trainer.step = state_dict["step"]
            trainer.model.load_state_dict(state_dict["model"])

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    logger.print("Testing forward...", end=" ", flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    loss, _ = bc.loss(*batch)
    loss.backward()
    logger.print("✓")

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#

    n_epochs = int((Config.n_train_steps - trainer.step) // Config.n_steps_per_epoch)

    for i in range(n_epochs):
        logger.print(f"Epoch {i} / {n_epochs} | {logger.prefix}")
        trainer.train(n_train_steps=Config.n_steps_per_epoch)
    trainer.finish_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)

    from ml_logger import RUN, logger

    Config = build_config_from_dict(exp_specs)

    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    job_name = Config.job_name.format(**vars(Config))
    RUN.prefix, RUN.job_name, _ = RUN(
        script_path=__file__,
        exp_name=exp_specs["exp_name"],
        job_name=job_name + f"/{Config.seed}",
    )

    logger.configure(RUN.prefix, root=RUN.script_root)
    # logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text(
        """
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: a0_loss
                      xKey: steps
                    """,
        filename=".charts.yml",
        dedent=True,
        overwrite=True,
    )
    logger.save_yaml(exp_specs, "exp_specs.yml")

    main(Config, RUN)
