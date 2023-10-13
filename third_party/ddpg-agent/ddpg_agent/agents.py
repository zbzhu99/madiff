from ddpg_agent.misc import hard_update
from ddpg_agent.networks import MLPNetwork
from torch.autograd import Variable
from torch.optim import Adam


class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target critic, exploration noise)
    """

    def __init__(
        self,
        num_in_pol,
        num_out_pol,
        num_in_critic,
        hidden_dim=64,
        lr=0.01,
        discrete_action=True,
        gaussian_noise_std=None,
    ):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        num_out_critic = 1

        self.policy = MLPNetwork(
            num_in_pol,
            num_out_pol,
            hidden_dim=hidden_dim,
            constrain_out=True,
            discrete_action=discrete_action,
        )
        self.target_policy = MLPNetwork(
            num_in_pol,
            num_out_pol,
            hidden_dim=hidden_dim,
            constrain_out=True,
            discrete_action=discrete_action,
        )
        self.critic = MLPNetwork(
            num_in_critic, num_out_critic, hidden_dim=hidden_dim, constrain_out=False
        )
        self.target_critic = MLPNetwork(
            num_in_critic, num_out_critic, hidden_dim=hidden_dim, constrain_out=False
        )

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        self.gaussian_noise_std = gaussian_noise_std

        self.discrete_action = discrete_action

    def to(self, device):
        self.policy = self.policy.to(device)
        self.target_policy = self.target_policy.to(device)
        self.critic = self.critic.to(device)
        self.target_critic = self.target_critic.to(device)

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if explore:
            x = action.clone().zero_()
            gaussian_noise = self.gaussian_noise_std * x.clone().normal_()
            action += Variable(gaussian_noise, requires_grad=False)
        action = action.clamp(-1, 1)
        return action

    def load_params_without_optims(self, params):
        self.policy.load_state_dict(params["policy"])
        self.critic.load_state_dict(params["critic"])
        self.target_policy.load_state_dict(params["target_policy"])
        self.target_critic.load_state_dict(params["target_critic"])
        self.policy_optimizer = None
        self.critic_optimizer = None
