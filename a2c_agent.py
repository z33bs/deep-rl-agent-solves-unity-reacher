import torch
import numpy as np
from buffer import Storage
from policy_models_v2 import GaussianActorCriticNet

from config import Config
import pickle
import torch.nn as nn
import logger
import collections

class A2CAgent:
    def __init__(self, config, env):
        self.config = config
        self.logger = logger.get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

        self.config = config
        self.env = env
        self.network = GaussianActorCriticNet(self.env.state_size, self.env.action_size, self.config.hidden_dim)
        # self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=0.0007)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4, eps=1.e-5)
        self.total_steps = 0
        self.states, _, _ = self.env.reset(train_mode=False)
        self.scores = np.zeros(self.env.num_agents)
        self.done = False
        self.storage = Storage()

        self.running_rewards = []  # collections.deque(500*[0], 500)
        self.rollout_reward_threshold = self.config.rollout_min_reward_start
        self.running_rollout_t = collections.deque([0]*25, 25)
        self.running_loss = collections.deque([0.] * 25, 25)

    def reset(self, train_mode=True):
        self.states, _, _ = self.env.reset(train_mode)
        self.scores = np.zeros(self.env.num_agents)
        self.rollout_reward_threshold += self.config.rollout_min_reward_incr

    def step(self):
        self.storage.reset()
        states = self.states

        t = 0
        self.running_rewards = []
        should_roll = t < self.config.rollout_length
        if self.config.use_rollout_reward_threshold:
            should_roll = should_roll or np.array(self.running_rewards).sum() < min(self.rollout_reward_threshold,
                                                       self.config.rollout_min_reward_max)
        while should_roll:
            prediction = self.network(states)  # 'action' 'log_pi_a' 'entropy' 'mean' 'v'
            rewards, next_states, terminals = self.env.transition(
                self.to_np(prediction['action']))  # list 20, ndarray 20,33, list 20 bool
            # rewards = rewards # no normalization
            self.storage.feed(prediction)
            self.storage.feed({'reward': self.tensor(rewards).unsqueeze(-1),
                               'mask': (1 - self.tensor(list(map(int, terminals)))).unsqueeze(-1)})

            states = next_states
            self.total_steps += self.env.num_agents

            self.scores += rewards
            self.running_rewards.append(np.array(rewards).mean())
            t += 1

        self.states = states
        prediction = self.network(states)  # 'action' 'log_pi_a' 'entropy' 'mean' 'v'

        advantages = self.tensor(np.zeros((self.env.num_agents, 1)))
        self.storage.advantage = [torch.zeros(self.env.num_agents, 1)]*len(self.storage.reward)
        self.storage.ret = [torch.zeros(self.env.num_agents, 1)]*len(self.storage.reward)

        if not np.any(terminals):  # if episode not complete estimate future rewards
            returns = prediction['v'].detach()
        else:
            returns = 0

        for i in reversed(range(t)):
            returns = self.storage.reward[i] + self.config.discount * self.storage.mask[i] * returns
            if not self.config.use_gae:
                advantages = returns - self.storage.v[i].detach()
            else:
                td_error = self.storage.reward[i] + self.config.discount * self.storage.mask[i] * prediction['v'] - \
                           self.storage.v[i]
                advantages = advantages * self.config.gae_tau * self.config.discount * self.storage.mask[i] + td_error
            self.storage.advantage[i] = advantages.detach()
            self.storage.ret[i] = returns.detach()

        entries = self.storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])  # tensor 100,1
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()
        loss = (policy_loss - self.config.entropy_weight * entropy_loss +
                self.config.value_loss_weight * value_loss)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        self.running_loss.append(loss.detach())
        self.running_rollout_t.append(t)
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
