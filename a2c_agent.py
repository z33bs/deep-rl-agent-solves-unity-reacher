import torch
import numpy as np
from buffer import Storage
from policy_models_v2 import GaussianActorCriticNet, FCBody
from unity_environment import Env
from config import Config
import pickle
import torch.nn as nn
import logger


class A2CAgent:
    def __init__(self, config):
        self.config = config
        self.logger = logger.get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

        self.config = config
        self.env = Env('reacher.app')
        self.network = GaussianActorCriticNet(self.env.state_size, self.env.action_size)
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=0.0007)
        self.total_steps = 0
        self.states, _, _ = self.env.reset(train_mode=False)

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states

        scores = np.zeros(self.env.num_agents)

        for t in range(config.rollout_length):
            prediction = self.network(states)  # 'action' 'log_pi_a' 'entropy' 'mean' 'v'
            rewards, next_states, terminals = self.env.transition(self.to_np(prediction['action'])) # list 20, ndarray 20,33, list 20 bool
            # self.record_online_return(info)
            # rewards = rewards # no normalization
            storage.feed(prediction)
            storage.feed({'reward': self.tensor(rewards).unsqueeze(-1),
                         'mask': (1 - self.tensor(list(map(int, terminals)))).unsqueeze(-1)})

            states = next_states
            self.total_steps += self.env.num_agents

            scores += rewards
            # if np.any(terminals) or t == config.rollout_length - 1:
            if t == config.rollout_length - 1:
            # print('Episode: {} Total score this episode: {} Last {} average: {}'
                #       .format(i + 1, last_mean_reward, min(i + 1, 100),last_average))
                # print('Episode: ?, rl {} Total score this episode: {}'
                #       .format(t, np.mean(scores)))
                ret = np.mean(scores)
                # self.logger.add_scalar('episodic_return_train', ret, self.total_steps)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps, ret))
                break

        self.states = states
        prediction = self.network(states)  # 'action' 'log_pi_a' 'entropy' 'mean' 'v'
        storage.feed(prediction)
        storage.placeholder()

        advantages = self.tensor(np.zeros((self.env.num_agents, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
        # for i in reversed(range(len(storage.reward))):
            returns = storage.reward[i] + config.discount * storage.mask[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy']) #tensor 100,1
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_episode(self):
        state , _, _ = self.env.reset(train_mode=False)
        scores = np.zeros(self.env.num_agents)
        while True:
            self.network.eval()
            prediction = self.network(state)  # 'action' 'log_pi_a' 'entropy' 'mean' 'v'
            self.network.train()
            reward, state, done = self.env.transition(self.to_np(prediction['action']))
            # state, reward, done, info = env.step(action)
            scores += reward
            # ret = info[0]['episodic_return']
            # if ret is not None:
            #     break
            if np.any(done):
                break
        return np.mean(scores)

    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        # self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
        }

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

