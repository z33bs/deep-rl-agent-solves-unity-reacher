import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.autograd import Variable

# From spinup... nicer implementation
# TODO
# preprocess input to tensor per v2
# output tensor per v2
# ensure the no_grad is not propegated
# instantiate both nets with seed and see if does the same thing?
# experiment with weight initialization - can start with v2 but maybe kaiming?

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        # obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        # if isinstance(action_space, Box):
        self.pi = MLPGaussianActor(obs_dim, action_dim, hidden_sizes, activation)
        # elif isinstance(action_space, Discrete):
        #     self.pi = MLPCategoricalActor(obs_dim, action_dim, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        # obs = Variable(torch.from_numpy(obs).float().unsqueeze(0))
        return self.step(obs)[0]


class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)
