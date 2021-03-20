#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
# from .normalizer import *
# import argparse
import torch
import numpy as np
# from policy_models_v2 import GaussianActorCriticNet, FCBody

class RescaleNormalizer:
    def __init__(self, coef=1.0, read_only=False):
        self.coef = coef
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x

class Config:
    DEVICE = torch.device('cpu')
    NOISY_LAYER_STD = 0.1
    # DEFAULT_REPLAY = 'replay'
    # PRIORITIZED_REPLAY = 'prioritized_replay'

    def __init__(self):
        # generate_tag(kwargs)
        # kwargs.setdefault('log_level', 0)
        # config = Config()
        # config.merge(kwargs)

        # self.num_workers = 16
        # config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
        # config.eval_env = Task(config.game)
        self.discount = 0.99
        self.use_gae = True
        self.gae_tau = 1.0
        self.entropy_weight = 0.01
        self.rollout_length = 5
        self.gradient_clip = 5
        self.max_steps = int(2e7)
        # run_steps(A2CAgent(config))


    #     self.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    #     self.network_fn = lambda: GaussianActorCriticNet(
    #         config.state_dim, config.action_dim,
    #         actor_body=FCBody(config.state_dim), critic_body=FCBody(config.state_dim))
    #     self.discount = 0.99
    #     self.use_gae = True
    #     self.gae_tau = 1.0
    #     self.entropy_weight = 0.01
    #     self.rollout_length = 5
    #     self.gradient_clip = 5
    #     self.max_steps = int(2e7)
    #
    #     # self.parser = argparse.ArgumentParser()
    #     # self.task_fn = None
    #     # self.actor_optimizer_fn = None
    #     # self.critic_optimizer_fn = None
    #     # self.actor_network_fn = None
    #     # self.critic_network_fn = None
    #     # self.replay_fn = None
    #     # self.random_process_fn = None
    #     # self.target_network_update_freq = None
    #     # self.exploration_steps = None
        self.log_level = 0
    #     # self.history_length = None
    #     # self.double_q = False
        self.tag = 'vanilla'
    #     # self.num_workers = 1
    #     # self.gradient_clip = None
    #     # self.target_network_mix = 0.001
    #     # self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
    #     # self.min_memory_size = None
    #     # self.max_steps = 0
        self.value_loss_weight = 1.0
        self.iteration_log_interval = 30
    #     # self.categorical_v_min = None
    #     # self.categorical_v_max = None
    #     # self.categorical_n_atoms = 51
    #     # self.num_quantiles = None
    #     # self.optimization_epochs = 4
    #     # self.mini_batch_size = 64
    #     # self.termination_regularizer = 0
    #     # self.sgd_update_frequency = None
    #     # self.random_action_prob = None
    #     # self.__eval_env = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
    #     # self.eval_episodes = 10
    #     # self.async_actor = True
        self.tasks = False
    #     # self.replay_type = Config.DEFAULT_REPLAY
    #     # self.decaying_lr = False
    #     # self.shared_repr = False
    #     # self.noisy_linear = False
    #     # self.n_step = 1
    #
    # # @property
    # # def eval_env(self):
    # #     return self.__eval_env
    # #
    # # @eval_env.setter
    # # def eval_env(self, env):
    # #     self.__eval_env = env
    # #     self.state_dim = env.state_dim
    # #     self.action_dim = env.action_dim
    # #     self.task_name = env.name
    # #
    # # def add_argument(self, *args, **kwargs):
    # #     self.parser.add_argument(*args, **kwargs)
    # #
    # # def merge(self, config_dict=None):
    # #     if config_dict is None:
    # #         args = self.parser.parse_args()
    # #         config_dict = args.__dict__
    # #     for key in config_dict.keys():
    # #         setattr(self, key, config_dict[key])