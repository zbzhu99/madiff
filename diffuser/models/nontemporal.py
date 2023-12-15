from numbers import Number

import numpy as np
import torch
import torch.nn as nn

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class ReparamMultivariateNormalDiag:
    """
    My reparameterized normal implementation
    """

    def __init__(self, mean, log_sig_diag):
        self.mean = mean
        self.log_sig_diag = log_sig_diag
        self.log_cov = 2.0 * log_sig_diag
        self.cov = torch.exp(self.log_cov)
        self.sig = torch.exp(self.log_sig_diag)
        self.device = mean.device

    def sample(self):
        eps = torch.randn(self.mean.size(), requires_grad=False)
        eps = eps.to(self.mean.device)
        samples = eps * self.sig + self.mean
        return samples

    def sample_n(self, n):
        # cleanly expand float or Tensor or Variable parameters
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v], requires_grad=False).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        expanded_mean = expand(self.mean)
        expanded_sig = expand(self.sig)
        eps = torch.randn(expanded_mean.size(), requires_grad=False)
        return eps * expanded_sig + expanded_mean

    def log_prob(self, value):
        assert value.dim() >= 2, "Where is the batch dimension?"

        log_prob = -0.5 * torch.sum(
            (self.mean - value) ** 2 / self.cov, -1, keepdim=True
        )
        rest = torch.sum(self.log_sig_diag, -1, keepdim=True) + 0.5 * np.log(2 * np.pi)
        log_prob -= rest
        return log_prob


class BCMLPnet(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        init_w=1e-3,
        conditioned_std: bool = False,
    ):
        super().__init__()

        dims = [observation_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/nontemporal ] MLP dimensions: {in_out}")

        act_fn = nn.Mish()

        self.conditioned_std = conditioned_std
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        mlp = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            mlp.extend([nn.Linear(dim_in, dim_out), act_fn])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            mlp.extend([nn.Linear(dim_out, dim_in), act_fn])

        self.mlp = nn.Sequential(*mlp)
        self.last_fc = nn.Linear(dim, action_dim)

        if self.conditioned_std:
            last_hidden_size = dim
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.last_fc.weight.data.mul_(0.1)
        self.last_fc.bias.data.mul_(0.0)

    def forward(self, obs, deterministic=False):
        """
        obs : [ batch x agent x obs_dim ]
        """

        h = self.mlp(obs)
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        if deterministic:
            action = mean
            log_prob = None
        else:
            normal = ReparamMultivariateNormalDiag(mean, log_std)
            action = normal.sample()
            log_prob = normal.log_prob(action)

        return action, mean, log_std, log_prob

    def get_log_prob(self, obs, acts):
        h = self.mlp(obs)
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        normal = ReparamMultivariateNormalDiag(mean, log_std)
        log_prob = normal.log_prob(acts)

        return log_prob
