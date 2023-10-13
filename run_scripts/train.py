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

    customized_kwargs = dict()
    if Config.env_type == "nba":
        customized_kwargs["nba_hz"] = Config.nba_hz
        customized_kwargs["nba_eval_valid_samples"] = Config.nba_eval_valid_samples

    dataset_config = utils.Config(
        Config.loader,
        savepath="dataset_config.pkl",
        env_type=Config.env_type,
        env=Config.dataset,
        n_agents=Config.n_agents,
        horizon=Config.horizon,
        history_horizon=Config.history_horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        max_n_episodes=Config.max_n_episodes,
        use_padding=Config.use_padding,
        use_action=Config.use_action,
        discrete_action=getattr(Config, "discrete_action", False),
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
        agent_share_parameters=utils.config.import_class(
            Config.model
        ).agent_share_parameters,
        **customized_kwargs,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath="render_config.pkl",
        env_type=Config.env_type,
        env=Config.dataset,
    )
    data_encoder_config = utils.Config(
        getattr(Config, "data_encoder", "utils.IdentityEncoder"),
        savepath="data_encoder_config.pkl",
    )

    dataset = dataset_config()
    renderer = render_config()
    data_encoder = data_encoder_config()
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    if Config.diffusion == "models.GaussianInvDynDiffusion":
        model_config = utils.Config(
            Config.model,
            savepath="model_config.pkl",
            n_agents=Config.n_agents,
            horizon=Config.horizon + Config.history_horizon,
            transition_dim=observation_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
            device=Config.device,
        )

        diffusion_config = utils.Config(
            Config.diffusion,
            savepath="diffusion_config.pkl",
            n_agents=Config.n_agents,
            horizon=Config.horizon + Config.history_horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            discrete_action=getattr(Config, "discrete_action", False),
            num_actions=getattr(dataset.env, "num_actions", 0),
            n_timesteps=Config.n_diffusion_steps,
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            predict_epsilon=Config.predict_epsilon,
            hidden_dim=Config.hidden_dim,
            ar_inv=Config.ar_inv,
            train_only_inv=Config.train_only_inv,
            share_inv=Config.share_inv,
            # loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            condition_guidance_w=Config.condition_guidance_w,
            agent_share_noise=Config.agent_share_noise,
            data_encoder=data_encoder,
            device=Config.device,
        )

    else:
        model_config = utils.Config(
            Config.model,
            savepath="model_config.pkl",
            n_agents=Config.n_agents,
            horizon=Config.horizon + Config.history_horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
            device=Config.device,
        )

        diffusion_config = utils.Config(
            Config.diffusion,
            savepath="diffusion_config.pkl",
            horizon=Config.horizon + Config.history_horizon,
            n_agents=Config.n_agents,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=Config.n_diffusion_steps,
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            predict_epsilon=Config.predict_epsilon,
            # loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            condition_guidance_w=Config.condition_guidance_w,
            agent_share_noise=Config.agent_share_noise,
            data_encoder=data_encoder,
            device=Config.device,
        )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath="trainer_config.pkl",
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        eval_freq=Config.eval_freq,
        save_parallel=Config.save_parallel,
        bucket=logger.root,
        n_reference=Config.n_reference,
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
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

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
            trainer.ema_model.load_state_dict(state_dict["ema"])

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    logger.print("Testing forward...", end=" ", flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    loss, _ = diffusion.loss(*batch)
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
    logger.remove("parameters.pkl")
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
