import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config
#
# DEVICE = torch.device('cpu')
# NOISY_LAYER_STD = 0.1

# From shandong

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(Config.DEVICE)
    return x


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        if noisy_linear:
            self.layers = nn.ModuleList(
                [NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList(
                [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


#GaussianActorCriticNet(
    # config.state_dim, config.action_dim,
    # actor_body=FCBody(config.state_dim), critic_body=FCBody(config.state_dim))
class GaussianActorCriticNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 # phi_body=None,
                 # actor_body=None,
                 # critic_body=None
                 ):
        super(GaussianActorCriticNet, self).__init__()
        # if phi_body is None: phi_body = DummyBody(state_dim)
        # if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        # if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        # self.phi_body = phi_body
        self.actor_body = FCBody(state_dim)
        self.critic_body = FCBody(state_dim)
        self.fc_action = layer_init(nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(self.critic_body.feature_dim, 1), 1e-3)
        self.std = nn.Parameter(torch.zeros(action_dim))
        # self.phi_params = list(self.phi_body.parameters())

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters()) #+ self.phi_params
        self.actor_params.append(self.std)
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters()) #+ self.phi_params

        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        # phi = self.phi_body(obs)
        phi_a = self.actor_body(obs)
        phi_v = self.critic_body(obs)
        mean = torch.tanh(self.fc_action(phi_a))
        v = self.fc_critic(phi_v) #20,1
        dist = torch.distributions.Normal(mean, F.softplus(self.std)) #batch shape 20,4

        if action is None:
            action = dist.sample() #20,4
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1) #20,1
        entropy = dist.entropy().sum(-1).unsqueeze(-1) #20,1
        return {'action': action, #20,4
                'log_pi_a': log_prob, #20,1
                'entropy': entropy, #20,1
                'mean': mean, #20,4
                'v': v} #20,1

    def layer_init(layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer


# Adapted from https://github.com/saj1919/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=True)
        self.weight_sigma = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=True)
        self.register_buffer('weight_epsilon', torch.zeros((out_features, in_features)))

        self.bias_mu = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.bias_sigma = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        self.register_buffer('noise_in', torch.zeros(in_features))
        self.register_buffer('noise_out_weight', torch.zeros(out_features))
        self.register_buffer('noise_out_bias', torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        self.noise_in.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_weight.normal_(std=Config.NOISY_LAYER_STD)
        self.noise_out_bias.normal_(std=Config.NOISY_LAYER_STD)

        self.weight_epsilon.copy_(self.transform_noise(self.noise_out_weight).ger(
            self.transform_noise(self.noise_in)))
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    def transform_noise(self, x):
        return x.sign().mul(x.abs().sqrt())
