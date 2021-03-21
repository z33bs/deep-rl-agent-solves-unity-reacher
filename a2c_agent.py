import torch
import numpy as np
from buffer import Storage
from policy_models_v2 import GaussianActorCriticNet

from config import Config
import pickle
import torch.nn as nn
import logger


class A2CAgent:
    def __init__(self, config, env):
        self.config = config
        self.logger = logger.get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

        self.config = config
        self.env = env
        self.network = GaussianActorCriticNet(self.env.state_size, self.env.action_size)
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=0.0007)
        self.total_steps = 0
        self.states, _, _ = self.env.reset(train_mode=False)
        self.scores = np.zeros(self.env.num_agents)
        self.done = False

    def reset(self, train_mode=True):
        self.states, _, _ = self.env.reset(train_mode)
        self.scores = np.zeros(self.env.num_agents)

    def step(self):
        storage = Storage()
        states = self.states

        for t in range(self.config.rollout_length):
            prediction = self.network(states)  # 'action' 'log_pi_a' 'entropy' 'mean' 'v'
            rewards, next_states, terminals = self.env.transition(
                self.to_np(prediction['action']))  # list 20, ndarray 20,33, list 20 bool
            # rewards = rewards # no normalization
            storage.feed(prediction)
            storage.feed({'reward': self.tensor(rewards).unsqueeze(-1),
                          'mask': (1 - self.tensor(list(map(int, terminals)))).unsqueeze(-1)})

            states = next_states
            self.total_steps += self.env.num_agents

            self.scores += rewards

        self.states = states
        prediction = self.network(states)  # 'action' 'log_pi_a' 'entropy' 'mean' 'v'
        # storage.placeholder()  # empty 'advantage' 'ret'

        advantages = self.tensor(np.zeros((self.env.num_agents, 1)))
        storage.advantage = [np.empty_like(advantages)] * len(storage.reward)
        storage.ret = [np.empty_like(advantages)] * len(storage.reward)

        if not np.any(terminals):  # if episode not complete estimate future rewards
            returns = prediction['v'].detach()
        else:
            returns = 0

        for i in reversed(range(self.config.rollout_length)):
            returns = storage.reward[i] + self.config.discount * storage.mask[i] * returns
            if not self.config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + self.config.discount * storage.mask[i] * prediction['v'] - storage.v[i]
                advantages = advantages * self.config.gae_tau * self.config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])  # tensor 100,1
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - self.config.entropy_weight * entropy_loss +
         self.config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        # test = np.any(terminals)
        # if test:
        #     found = True
        return np.any(terminals)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    # def record_online_return(self, info, offset=0):
    #     if isinstance(info, dict):
    #         ret = info['episodic_return']
    #         if ret is not None:
    #             self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
    #             self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
    #     elif isinstance(info, tuple):
    #         for i, info_ in enumerate(info):
    #             self.record_online_return(info_, i)
    #     else:
    #         raise NotImplementedError

    def tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        x = np.asarray(x, dtype=np.float32)
        x = torch.from_numpy(x).to(Config.DEVICE)
        return x

    def to_np(self, t):
        return t.cpu().detach().numpy()
