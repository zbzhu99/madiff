import functools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import diffuser.utils as utils
from diffuser.utils.dpm_solver import DPM_Solver, NoiseScheduleVP, model_wrapper

from .helpers import Losses, apply_conditioning, cosine_beta_schedule, extract


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        n_agents,
        horizon,
        history_horizon,
        observation_dim,
        action_dim,
        n_timesteps=1000,
        loss_type="l1",
        clip_denoised=False,
        predict_epsilon=True,
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
        returns_condition=False,
        condition_guidance_w=0.1,
        agent_share_noise=False,
        data_encoder=utils.IdentityEncoder(),
        **kwargs,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.agent_share_noise = agent_share_noise
        self.data_encoder = data_encoder

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # get loss coefficients and initialize objective
        self.loss_type = loss_type
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        sets loss coefficients for trajectory

        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c
        """
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(
            self.horizon + self.history_horizon, dtype=torch.float
        )
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights = loss_weights.unsqueeze(1).expand(-1, self.n_agents, -1).clone()

        # manually set a0 weight
        loss_weights[self.history_horizon, :, : self.action_dim] = action_weight
        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, t, returns=returns, use_dropout=False)
            epsilon_uncond = self.model(x, t, returns=returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )
        else:
            epsilon = self.model(x, t)

        t = t.detach().to(torch.int64)
        if (
            "player_idxs" in cond and "player_hoop_sides" in cond
        ):  # must have add player info
            x = x[:, :, :, 1:-1]
            x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, returns=returns
        )
        x = utils.remove_player_info(x, cond)
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self, shape, cond, returns=None, verbose=True, return_diffusion=False
    ):
        device = self.betas.device

        batch_size = shape[0]
        # low temperature sampling; alpha equals 0.5
        if self.agent_share_noise:
            x = 0.5 * torch.randn((shape[0], shape[1], shape[3]), device=device)
            x = torch.stack([x for _ in range(shape[2])], dim=2)
        else:
            x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)
        x = self.data_encoder(x)

        if return_diffusion:
            diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim)
            x = self.data_encoder(x)

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)
        x = utils.remove_player_info(x, cond)
        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        """
        conditions : [ (time, state), ... ]
        """

        batch_size = len(list(cond.values())[0])
        horizon = horizon or self.horizon + self.history_horizon
        shape = (batch_size, horizon, self.n_agents, self.transition_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, returns=returns
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(
        self, shape, cond, returns=None, verbose=True, return_diffusion=False
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)
        x = self.data_encoder(x)

        if return_diffusion:
            diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.grad_p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim)
            x = self.data_encoder(x)

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(
        self, cond, returns=None, horizon=None, *args, **kwargs
    ):
        """
        conditions : [ (time, state), ... ]
        """

        batch_size = len(list(cond.values())[0])
        horizon = horizon or self.horizon + self.history_horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.grad_p_sample_loop(shape, cond, returns, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        if self.agent_share_noise:
            noise = torch.randn_like(x_start[:, :, 0])
            noise = torch.stack([noise for _ in range(x_start.shape[2])], dim=2)
        else:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
        x_noisy = self.data_encoder(x_noisy)

        x_recon = self.model(x_noisy, t, returns=returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim)
            if (
                "player_idxs" in cond and "player_hoop_sides" in cond
            ):  # must have add player info
                x_recon = x_recon[:, :, :, 1:-1]  # except player info
            x_recon = self.data_encoder(x_recon)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, masks=None, returns=None, states=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffuse_loss, info = self.p_losses(x, cond, t, returns)
        diffuse_loss = (diffuse_loss * masks.unsqueeze(-1)).mean()
        return diffuse_loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)


class GaussianInvDynDiffusion(nn.Module):
    def __init__(
        self,
        model,
        n_agents: int,
        horizon: int,
        history_horizon: int,
        observation_dim: int,
        action_dim: int,
        state_dim: int,
        use_state: bool = False,
        discrete_action: bool = False,
        num_actions: int = 0,  # for discrete action space
        n_timesteps: int = 1000,
        loss_type: str = "l1",
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        hidden_dim: int = 256,
        action_weight: float = 1.0,
        loss_discount: float = 1.0,
        loss_weights: np.ndarray = None,
        state_loss_weight: float = None,
        opponent_loss_weight: float = None,
        returns_condition: bool = False,
        condition_guidance_w: float = 1.2,
        returns_loss_guided: bool = False,
        returns_loss_clean_only: bool = False,
        loss_guidence_w: float = 0.1,
        value_diffusion_model: nn.Module = None,
        ar_inv: bool = False,
        train_only_inv: bool = False,
        share_inv: bool = True,
        joint_inv: bool = False,
        agent_share_noise: bool = False,
        data_encoder: utils.Encoder = utils.IdentityEncoder(),
        **kwargs,
    ):
        assert action_dim > 0
        assert (
            not returns_condition or not returns_loss_guided
        ), "Can't do both returns conditioning and returns loss guidence"
        # if returns_loss_guided:
        #     assert (
        #         value_diffusion_model is not None
        #     ), "Must provide value diffusion model when using returns loss guidence"

        super().__init__()
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.state_loss_weight = state_loss_weight
        self.opponent_loss_weight = opponent_loss_weight
        self.use_state = use_state
        self.discrete_action = discrete_action
        self.num_actions = num_actions
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        self.share_inv = share_inv
        self.joint_inv = joint_inv
        self.agent_share_noise = agent_share_noise
        self.data_encoder = data_encoder

        self.inv_model = self._build_inv_model(
            hidden_dim, output_dim=action_dim if not discrete_action else num_actions
        )

        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        self.returns_loss_guided = returns_loss_guided
        self.returns_loss_clean_only = returns_loss_clean_only
        self.loss_guidence_w = loss_guidence_w
        self.value_diffusion_model = value_diffusion_model
        if self.value_diffusion_model is not None:
            self.value_diffusion_model.requires_grad_(False)

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses["state_l2"](loss_weights)
        if self.use_state:
            state_loss_weights = self.get_state_loss_weights(loss_discount)
            self.state_loss_fn = Losses["state_l2"](state_loss_weights)

        self.dpm_solver = None

    def _build_inv_model(self, hidden_dim: int, output_dim: int):
        if self.joint_inv:
            print("\nUSE JOINT INV\n")
            inv_model = nn.Sequential(
                nn.Linear(self.n_agents * (2 * self.observation_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_agents * output_dim),
            )

        elif self.share_inv:
            print("\nUSE SHARED INV\n")
            inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        else:
            print("\nUSE INDEPENDENT INV\n")
            inv_model = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * self.observation_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, output_dim),
                        nn.Softmax(dim=-1) if self.discrete_action else nn.Identity(),
                    )
                    for _ in range(self.n_agents)
                ]
            )

        return inv_model

    def get_state_loss_weights(self, discount: float):
        dim_weights = torch.ones(self.state_dim, dtype=torch.float32)

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights = loss_weights.clone()

        return loss_weights

    def get_loss_weights(self, discount: float):
        """
        sets loss coefficients for trajectory

        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c
        """

        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = torch.cat([torch.zeros(self.history_horizon), discounts])
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights = loss_weights.unsqueeze(1).expand(-1, self.n_agents, -1).clone()

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        x,
        t,
        returns=None,
        env_ts=None,
        attention_masks=None,
        states=None,
        return_xstart=False,
    ):
        if self.use_state:
            assert states is not None
            if self.returns_condition:
                # epsilon could be epsilon or x0 itself
                epsilon_cond, state_epsilon_cond = self.model(
                    x,
                    t,
                    returns=returns,
                    env_timestep=env_ts,
                    attention_masks=attention_masks,
                    states=states,
                    use_dropout=False,
                )
                epsilon_uncond, state_epsilon_uncond = self.model(
                    x,
                    t,
                    returns=returns,
                    env_timestep=env_ts,
                    attention_masks=attention_masks,
                    states=states,
                    force_dropout=True,
                )
                epsilon = epsilon_uncond + self.condition_guidance_w * (
                    epsilon_cond - epsilon_uncond
                )
                state_epsilon = state_epsilon_uncond + self.condition_guidance_w * (
                    state_epsilon_cond - state_epsilon_uncond
                )
            else:
                epsilon, state_epsilon = self.model(
                    x,
                    t,
                    env_timestep=env_ts,
                    attention_masks=attention_masks,
                    states=states,
                )
        else:
            if self.returns_condition:
                # epsilon could be epsilon or x0 itself
                epsilon_cond = self.model(
                    x,
                    t,
                    returns=returns,
                    env_timestep=env_ts,
                    attention_masks=attention_masks,
                    use_dropout=False,
                )
                epsilon_uncond = self.model(
                    x,
                    t,
                    returns=returns,
                    env_timestep=env_ts,
                    attention_masks=attention_masks,
                    force_dropout=True,
                )
                epsilon = epsilon_uncond + self.condition_guidance_w * (
                    epsilon_cond - epsilon_uncond
                )
            else:
                epsilon = self.model(
                    x, t, env_timestep=env_ts, attention_masks=attention_masks
                )

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        if self.use_state:
            state_recon = self.predict_start_from_noise(states, t, noise=state_epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
            if self.use_state:
                state_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        if self.use_state:
            (
                state_model_mean,
                state_posterior_variance,
                state_posterior_log_variance,
            ) = self.q_posterior(x_start=state_recon, x_t=states, t=t)
            if return_xstart:
                return (
                    model_mean,
                    posterior_variance,
                    posterior_log_variance,
                    x_recon,
                    state_model_mean,
                    state_posterior_variance,
                    state_posterior_log_variance,
                )
            else:
                return (
                    model_mean,
                    posterior_variance,
                    posterior_log_variance,
                    state_model_mean,
                    state_posterior_variance,
                    state_posterior_log_variance,
                )

        else:
            if return_xstart:
                return model_mean, posterior_variance, posterior_log_variance, x_recon
            else:
                return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self, x, t, returns=None, env_ts=None, attention_masks=None, states=None
    ):
        b = x.shape[0]
        if self.use_state:
            (
                model_mean,
                _,
                model_log_variance,
                state_model_mean,
                _,
                state_model_log_variance,
            ) = self.p_mean_variance(
                x=x,
                t=t,
                returns=returns,
                env_ts=env_ts,
                attention_masks=attention_masks,
                states=states,
            )
        else:
            model_mean, _, model_log_variance = self.p_mean_variance(
                x=x,
                t=t,
                returns=returns,
                env_ts=env_ts,
                attention_masks=attention_masks,
            )

        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if self.use_state:
            state_noise = 0.5 * torch.randn_like(states)
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                state_model_mean + (0.5 * state_model_log_variance).exp() * state_noise,
            )
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        returns=None,
        env_ts=None,
        attention_masks=None,
        verbose=True,
        return_diffusion=False,
    ):
        device = self.betas.device

        batch_size = shape[0]
        if self.agent_share_noise:
            x = 0.5 * torch.randn((shape[0], shape[1], shape[3]), device=device)
            x = torch.stack([x for _ in range(shape[2])], dim=2)
        else:
            x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, 0)
        x = self.data_encoder(x)

        if self.use_state:
            state_shape = (batch_size, shape[1], self.state_dim)
            states = 0.5 * torch.randn(state_shape, device=device)

        if return_diffusion:
            diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if self.use_state:
                x, states = self.p_sample(
                    x, timesteps, returns, env_ts, attention_masks, states
                )
            else:
                x = self.p_sample(x, timesteps, returns, env_ts, attention_masks)
            x = apply_conditioning(x, cond, 0)
            x = self.data_encoder(x)

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(
        self,
        cond,
        returns=None,
        env_ts=None,
        horizon=None,
        attention_masks=None,
        use_dpm_solver=False,
        use_ddim_sample=False,
        *args,
        **kwargs,
    ):
        """
        conditions : [ (time, state), ... ]
        """

        batch_size = len(list(cond.values())[0])
        horizon = horizon or self.horizon + self.history_horizon
        shape = (batch_size, horizon, self.n_agents, self.observation_dim)

        # BUG(zbzhu): Dpm solver now samples very large values
        # TODO(mhliu): Dpm solver does not use data encoder
        if use_dpm_solver:
            assert not use_ddim_sample
            raise NotImplementedError
            if self.dpm_solver is None:
                noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)
                model_fn = model_wrapper(
                    self.model,
                    noise_schedule,
                    model_type="noise",
                )
                self.dpm_solver = DPM_Solver(
                    model_fn, noise_schedule, algorithm_type="dpmsolver++"
                )

            x = 0.5 * torch.randn(shape, device=self.betas.device)
            x = self.dpm_solver.sample(
                x,
                condition_func=functools.partial(
                    apply_conditioning, conditions=cond, action_dim=0
                ),
                steps=20,
                order=3,
                skip_type="time_uniform",
                method="singlestep",
            )
            return x

        elif use_ddim_sample:
            return self.ddim_sample_loop(
                shape, cond, returns, env_ts, attention_masks, *args, **kwargs
            )
        else:
            return self.p_sample_loop(
                shape, cond, returns, env_ts, attention_masks, *args, **kwargs
            )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    @torch.no_grad()
    def ddim_sample(
        self,
        x,
        cond,
        t,
        returns=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """

        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            x=x, cond=cond, t=t, returns=returns, return_xstart=True
        )

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, pred_xstart)

        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = 0.5 * torch.randn_like(x)
        mean_pred = (
            pred_xstart * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        # return {"sample": sample, "pred_xstart": out["pred_xstart"]}
        return sample

    @torch.no_grad()
    def ddim_reverse_sample(
        self,
        x,
        cond,
        t,
        returns=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """

        assert eta == 0.0, "Reverse ODE only for deterministic path"
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            x=x, cond=cond, t=t, returns=returns, return_xstart=True
        )

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            pred_xstart * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )
        return mean_pred

    @torch.no_grad()
    def ddim_sample_loop(
        self, shape, cond, returns=None, verbose=True, return_diffusion=False, eta=0.0
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """

        device = self.betas.device

        batch_size = shape[0]
        if self.agent_share_noise:
            x = 0.5 * torch.randn((shape[0], shape[1], shape[3]), device=device)
            x = torch.stack([x for _ in range(shape[2])], dim=2)
        else:
            x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, 0)
        x = self.data_encoder(x)

        if return_diffusion:
            diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.ddim_sample(x, cond, timesteps, returns, eta=eta)
            x = apply_conditioning(x, cond, 0)
            x = self.data_encoder(x)

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(
        self,
        x_start,
        cond,
        t,
        loss_masks,
        attention_masks=None,
        returns=None,
        env_ts=None,
        states=None,
    ):
        if self.agent_share_noise:
            print("\n\n!!! AGENT SHARE NOISE !!!\n\n")
            noise = torch.randn_like(x_start[:, :, 0])
            noise = torch.stack([noise for _ in range(x_start.shape[2])], dim=2)
        else:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, 0)
        x_noisy = self.data_encoder(x_noisy)

        if self.use_state:
            state_noise = torch.randn_like(states)
            states_noisy = self.q_sample(x_start=states, t=t, noise=state_noise)
            epsilon, state_epsilon = self.model(
                x_noisy,
                t,
                returns=returns,
                env_timestep=env_ts,
                states=states_noisy,
                attention_masks=attention_masks,
            )
        else:
            epsilon = self.model(
                x_noisy,
                t,
                returns=returns,
                env_timestep=env_ts,
                attention_masks=attention_masks,
            )

        if not self.predict_epsilon:
            epsilon = apply_conditioning(epsilon, cond, 0)
            epsilon = self.data_encoder(epsilon)

        assert noise.shape == epsilon.shape
        if self.use_state:
            assert state_noise.shape == state_epsilon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(epsilon, noise)
        else:
            loss, info = self.loss_fn(epsilon, x_start)

        if "agent_idx" in cond.keys() and self.opponent_loss_weight is not None:
            opponent_loss_weight = torch.ones_like(loss) * self.opponent_loss_weight
            indices = (
                cond["agent_idx"]
                .to(torch.long)[..., None]
                .repeat(
                    1, opponent_loss_weight.shape[1], 1, opponent_loss_weight.shape[-1]
                )
            )
            opponent_loss_weight.scatter_(dim=2, index=indices, value=1)
            loss = loss * opponent_loss_weight

        # TODO(zbzhu): Check these two '.mean()'
        loss = (
            (loss * loss_masks).mean(dim=[1, 2]) / loss_masks.mean(dim=[1, 2])
        ).mean()

        if self.use_state:
            if self.predict_epsilon:
                state_loss, _ = self.state_loss_fn(state_epsilon, state_noise)
            else:
                state_loss, _ = self.state_loss_fn(state_epsilon, states)
            state_loss = (state_loss * loss_masks[:, :, 0]).mean()
            info["state_loss"] = state_loss
            if self.state_loss_weight is not None:
                state_loss = state_loss * self.state_loss_weight
            # normalize state_loss by `n_agents`, otherwise it will be more important
            # as `n_agents` grows
            loss = loss + state_loss / self.n_agents

        if self.returns_loss_guided:
            returns_loss = self.r_losses(x_noisy, t, epsilon, cond)
            info["returns_loss"] = returns_loss
            loss = loss + returns_loss * self.loss_guidence_w

        return loss, info

    def r_losses(self, x_t, t, noise, cond):
        b = x_t.shape[0]
        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x_t, t, noise)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        if self.returns_loss_clean_only:
            value_pred = self.value_diffusion_model(x_t, torch.zeros_like(t))

        else:
            model_mean, _, model_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x_t, t=t
            )

            noise = 0.5 * torch.randn_like(x_t)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float()).reshape(
                b, *((1,) * (len(x_t.shape) - 1))
            )

            x_t_minus_1 = (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            )
            x_t_minus_1 = apply_conditioning(x_t_minus_1, cond, 0)
            x_t_minus_1 = self.data_encoder(x_t_minus_1)

            # in value_diffusion_model, t is trained as t - 1
            value_pred = self.value_diffusion_model(x_t_minus_1, t)

        # value_pred = torch.clamp(value_pred, 0.0, 400.0)
        return -1.0 * value_pred.mean()  # maximize value

    def loss(
        self,
        x,
        cond,
        loss_masks,
        attention_masks=None,
        returns=None,
        env_ts=None,
        states=None,
        legal_actions=None,
    ):
        if self.train_only_inv:
            info = {}
        else:
            batch_size = len(x)
            t = torch.randint(
                0, self.n_timesteps, (batch_size,), device=x.device
            ).long()
            diffuse_loss, info = self.p_losses(
                x[:, :, :, self.action_dim :],
                cond,
                t,
                loss_masks,
                attention_masks,
                returns,
                env_ts,
                states,
            )

        # Calculating inv loss
        x_t = x[:, :-1, :, self.action_dim :]
        a_t = x[:, :-1, :, : self.action_dim]
        x_t_1 = x[:, 1:, :, self.action_dim :]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, x_comb_t.shape[2], 2 * self.observation_dim)
        a_t = a_t.reshape(-1, a_t.shape[2], self.action_dim)
        masks_t = loss_masks[:, 1:].reshape(-1, loss_masks.shape[2])
        if legal_actions is not None:
            legal_actions_t = legal_actions[:, :-1].reshape(
                -1, *legal_actions.shape[2:]
            )

        if self.ar_inv:
            if self.share_inv:
                inv_loss = self.inv_model.calc_loss(x_comb_t, a_t)
            else:
                inv_loss = 0.0
                for i in range(self.n_agents):
                    inv_loss += self.inv_model[i].calc_loss(x_comb_t[:, i], a_t[:, i])

        else:
            if self.joint_inv or self.share_inv:
                if self.joint_inv:
                    pred_a_t = self.inv_model(
                        x_comb_t.reshape(x_comb_t.shape[0], -1)  # (b a) f
                    ).reshape(x_comb_t.shape[0], x_comb_t.shape[1], -1)
                else:
                    pred_a_t = self.inv_model(x_comb_t)

                if legal_actions is not None:
                    pred_a_t[legal_actions_t == 0] = -1e10
                if self.discrete_action:
                    inv_loss = (
                        F.cross_entropy(
                            pred_a_t.reshape(-1, pred_a_t.shape[-1]),
                            a_t.reshape(-1).long(),
                            reduction="none",
                        )
                        * masks_t.reshape(-1)
                    ).mean() / masks_t.mean()
                    inv_acc = (
                        (pred_a_t.argmax(dim=-1, keepdim=True) == a_t)
                        .to(dtype=float)
                        .squeeze(-1)
                        * masks_t
                    ).mean() / masks_t.mean()
                    info["inv_acc"] = inv_acc
                else:
                    inv_loss = (
                        F.mse_loss(pred_a_t, a_t, reduction="none")
                        * masks_t.unsqueeze(-1)
                    ).mean() / masks_t.mean()

            else:
                inv_loss = 0.0
                for i in range(self.n_agents):
                    pred_a_t = self.inv_model[i](x_comb_t[:, i])
                    if self.discrete_action:
                        inv_loss += (
                            F.cross_entropy(
                                pred_a_t, a_t[:, i].reshape(-1).long(), reduction="none"
                            )
                            * masks_t[:, i]
                        ).mean() / masks_t[:, i].mean()
                    else:
                        inv_loss += (
                            F.mse_loss(pred_a_t, a_t[:, i])
                            * masks_t[:, i].unsqueeze(-1)
                        ).mean() / masks_t[:, i].mean()

        info["inv_loss"] = inv_loss
        if self.train_only_inv:
            return inv_loss, info

        loss = (1 / 2) * (diffuse_loss + inv_loss)
        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)


class ARInvModel(nn.Module):
    def __init__(
        self, hidden_dim, observation_dim, action_dim, low_act=-1.0, up_act=1.0
    ):
        super(ARInvModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.action_embed_hid = 128
        self.out_lin = 128
        self.num_bins = 80

        self.up_act = up_act
        self.low_act = low_act
        self.bin_size = (self.up_act - self.low_act) / self.num_bins
        self.ce_loss = nn.CrossEntropyLoss()

        self.state_embed = nn.Sequential(
            nn.Linear(2 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lin_mod = nn.ModuleList(
            [nn.Linear(i, self.out_lin) for i in range(1, self.action_dim)]
        )
        self.act_mod = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, self.action_embed_hid),
                    nn.ReLU(),
                    nn.Linear(self.action_embed_hid, self.num_bins),
                )
            ]
        )

        for _ in range(1, self.action_dim):
            self.act_mod.append(
                nn.Sequential(
                    nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid),
                    nn.ReLU(),
                    nn.Linear(self.action_embed_hid, self.num_bins),
                )
            )

    def forward(self, comb_state, deterministic=False):
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(
                self.low_act + l_0 * self.bin_size,
                self.low_act + (l_0 + 1) * self.bin_size,
            ).sample()

        a = [a_0.unsqueeze(1)]

        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](
                torch.cat([state_d, self.lin_mod[i - 1](torch.cat(a, dim=1))], dim=1)
            )
            l_i = torch.distributions.Categorical(logits=lp_i).sample()

            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(
                    self.low_act + l_i * self.bin_size,
                    self.low_act + (l_i + 1) * self.bin_size,
                ).sample()

            a.append(a_i.unsqueeze(1))

        return torch.cat(a, dim=1)

    def calc_loss(self, comb_state, action):
        eps = 1e-8
        action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
        l_action = torch.div(
            (action - self.low_act), self.bin_size, rounding_mode="floor"
        ).long()
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

        for i in range(1, self.action_dim):
            loss += self.ce_loss(
                self.act_mod[i](
                    torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)
                ),
                l_action[:, i],
            )

        return loss / self.action_dim


class ActionGaussianDiffusion(nn.Module):
    # Assumes horizon=1
    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        n_timesteps=1000,
        loss_type="l1",
        clip_denoised=False,
        predict_epsilon=True,
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
        returns_condition=False,
        condition_guidance_w=0.1,
        **kwargs,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, t, returns=returns, use_dropout=False)
            epsilon_uncond = self.model(x, t, returns=returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )
        else:
            epsilon = self.model(x, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, returns=returns
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self, shape, cond, returns=None, verbose=True, return_diffusion=False
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)

        if return_diffusion:
            diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, *args, **kwargs):
        """
        conditions : [ (time, state), ... ]
        """

        raise NotImplementedError
        batch_size = len(list(cond.values())[0])
        shape = (batch_size, self.action_dim)
        cond = cond[0]  # FIXME(zbzhu)
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, returns=returns
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(
        self, shape, cond, returns=None, verbose=True, return_diffusion=False
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)

        if return_diffusion:
            diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, *args, **kwargs):
        """
        conditions : [ (time, state), ... ]
        """

        raise NotImplementedError
        batch_size = len(list(cond.values())[0])
        shape = (batch_size, self.action_dim)
        cond = cond[0]  # FIXME(zbzhu)
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, action_start, state, t, returns=None):
        noise = torch.randn_like(action_start)
        action_noisy = self.q_sample(x_start=action_start, t=t, noise=noise)

        pred = self.model(action_noisy, state, t, returns)

        assert noise.shape == pred.shape

        if self.predict_epsilon:
            loss = F.mse_loss(pred, noise)
        else:
            loss = F.mse_loss(pred, action_start)

        return loss, {"a0_loss": loss}

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        assert x.shape[1] == 1  # Assumes horizon=1
        x = x[:, 0, :]
        cond = x[:, self.action_dim :]  # Observation
        x = x[:, : self.action_dim]  # Action
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)


class ValueDiffusion(GaussianDiffusion):
    def __init__(self, *args, clean_only=False, agent_share_noise=False, **kwargs):
        assert "value" in kwargs["loss_type"]
        super().__init__(*args, **kwargs)
        if clean_only:
            print("[ models/diffusion ] Info: Only train on clean samples!")
        self.clean_only = clean_only
        self.agent_share_noise = agent_share_noise
        self.sqrt_alphas_cumprod = torch.cat(
            [
                torch.ones(1, device=self.betas.device),
                torch.sqrt(self.alphas_cumprod[:-1]),
            ]
        )
        self.sqrt_one_minus_alphas_cumprod = torch.cat(
            [
                torch.zeros(1, device=self.betas.device),
                torch.sqrt(1 - self.alphas_cumprod[:-1]),
            ]
        )

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        value_loss, info = self.p_losses(x, cond, returns, t - 1)
        value_loss = value_loss.mean()
        return value_loss, info

    def p_losses(self, x_start, cond, target, t):
        if self.clean_only:
            pred = self.model(x_start, torch.zeros_like(t))

        else:
            t = t + 1
            if self.agent_share_noise:
                noise = torch.randn_like(x_start[:, :, 0])
                noise = torch.stack([noise for _ in range(x_start.shape[2])], dim=2)
            else:
                noise = torch.randn_like(x_start)

            # since self.sqrt_alphas_cumprod and xxx is changed in __init__(),
            # x_noisy here is x_t_minus_1
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
            x_noisy = self.data_encoder(x_noisy)
            pred = self.model(x_noisy, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, t):
        return self.model(x, t)
