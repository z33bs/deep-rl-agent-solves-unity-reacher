import torch
import numpy as np
# from ..utils import *
# import torch.multiprocessing as mp
# from collections import deque
# from skimage.io import imsave

# from ..network import *
# from ..component import *
# from .BaseAgent import *

from buffer import Storage
from policy_models_v2 import GaussianActorCriticNet, FCBody
from unity_environment import Env
from config import Config
import pickle
import torch.nn as nn
import logger

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = logger.get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

    def close_obj(obj):
        if hasattr(obj, 'close'):
            obj.close()

    def close(self):
        self.close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    # def eval_step(self, state):
    #     raise NotImplementedError

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret

    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
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

    def switch_task(self):
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    # def record_episode(self, dir, env):
    #     mkdir(dir)
    #     steps = 0
    #     state = env.reset()
    #     while True:
    #         self.record_obs(env, dir, steps)
    #         action = self.record_step(state)
    #         state, reward, done, info = env.step(action)
    #         ret = info[0]['episodic_return']
    #         steps += 1
    #         if ret is not None:
    #             break
    #
    # def record_step(self, state):
    #     raise NotImplementedError
    #
    # # For DMControl
    # def record_obs(self, env, dir, steps):
    #     env = env.env.envs[0]
    #     obs = env.render(mode='rgb_array')
    #     imsave('%s/%04d.png' % (dir, steps), obs)


class A2CAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        # self.task = config.task_fn()
        self.env = Env('reacher.app')
        # self.network = config.network_fn()
        # self.optimizer = config.optimizer_fn(self.network.parameters())

        self.network = GaussianActorCriticNet(
            self.env.state_size, self.env.action_size)
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=0.0007)
        self.total_steps = 0
        self.states, _, _ = self.env.reset(train_mode=False)  # self.task.reset()

    def tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        x = np.asarray(x, dtype=np.float32)
        x = torch.from_numpy(x).to(Config.DEVICE)
        return x

    def to_np(self, t):
        return t.cpu().detach().numpy()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            # prediction = self.network(config.state_normalizer(states))
            prediction = self.network(states)
            pa = prediction['action'] #.cpu().detach().numpy()
            pa = self.to_np(pa)
            # next_states, rewards, terminals, info = self.env.transition(pa)  # self.to_np(prediction['action']))
            rewards, next_states, terminals = self.env.transition(self.to_np(prediction['action']))
            # list 20, ndarray 20,33, list 20 bool
            # self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.feed(prediction)
            storage.feed({'reward': self.tensor(rewards).unsqueeze(-1),
                         'mask': (1 - self.tensor(list(map(int, terminals)))).unsqueeze(-1)})

            states = next_states
            self.total_steps += self.env.num_agents #config.num_workers

        self.states = states
        prediction = self.network(states)
        storage.feed(prediction)
        storage.placeholder()

        advantages = self.tensor(np.zeros((self.env.num_agents, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.reward[i] + config.discount * storage.mask[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()