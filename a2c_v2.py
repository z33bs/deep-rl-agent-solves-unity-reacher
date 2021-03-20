# Todo
# implement as-is first
# then trash storage which is unnecessary
# then look if can use clipping function for loss? Probably not necessary as diverges
# modularize in my own way

from config import Config
import time
import os
import policy_models_v2
from unity_environment import Env
from a2c_agent import A2CAgent
from pathlib import Path
import numpy as np
import torch

# run steps def run_steps(agent):
def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
def random_seed(seed=None):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))
def select_device(gpu_id):
    # if torch.cuda.is_available() and gpu_id >= 0:
    if gpu_id >= 0:
        Config.DEVICE = torch.device('cuda:%d' % (gpu_id))
    else:
        Config.DEVICE = torch.device('cpu')
def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
mkdir('log')
mkdir('tf_log')
set_one_thread()
random_seed(333)
# -1 is CPU, a positive integer is the index of GPU
select_device(-1)

config = Config()
agent = A2CAgent(config)
agent_name = agent.__class__.__name__
t0 = time.time()
while True:
    if config.save_interval and not agent.total_steps % config.save_interval:
        agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
    if config.log_interval and not agent.total_steps % config.log_interval:
        agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
        t0 = time.time()
    if config.eval_interval and not agent.total_steps % config.eval_interval:
        agent.eval_episodes()
    if config.max_steps and agent.total_steps >= config.max_steps:
        agent.close()
        break
    agent.step()
    agent.switch_task()

# agent_name = 'test1'
# config = Config()
# env = Env('reacher.app')
#
# t0 = time.time()
#
# #BaseAgent.__init__(self, config)
# #self.task = config.task_fn()
# network = config.network_fn()
# optimizer = config.optimizer_fn(self.network.parameters())
# total_steps = 0
# # states, num_agents, (state_size, action_size) = env.reset()
# states, _, _ = env.reset()
#
# agent = A2CAgent(config)
#
# while True:
#
#     if config.save_interval and not total_steps % config.save_interval:
#         save('data/%s-%s-%d' % (agent_name, config.tag, total_steps))
#     if config.log_interval and not agent.total_steps % config.log_interval:
#         agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
#         t0 = time.time()
#     if config.eval_interval and not agent.total_steps % config.eval_interval:
#         agent.eval_episodes()
#     if config.max_steps and agent.total_steps >= config.max_steps:
#         agent.close()
#         break
#
#     # Agent step
#     config = self.config
#     storage = Storage(config.rollout_length)
#     states = self.states
#     # collect trajectories
#     for _ in range(config.rollout_length):
#         prediction = self.network(config.state_normalizer(states))
#         next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
#         self.record_online_return(info)
#         rewards = config.reward_normalizer(rewards)
#         storage.feed(prediction)
#         storage.feed({'reward': tensor(rewards).unsqueeze(-1),
#                       'mask': tensor(1 - terminals).unsqueeze(-1)})
#
#         states = next_states
#         self.total_steps += config.num_workers
#
#     self.states = states
#     prediction = self.network(config.state_normalizer(states))
#     storage.feed(prediction)
#     storage.placeholder()
#
#     advantages = tensor(np.zeros((config.num_workers, 1)))
#     returns = prediction['v'].detach()
#     for i in reversed(range(config.rollout_length)):
#         returns = storage.reward[i] + config.discount * storage.mask[i] * returns
#         if not config.use_gae:
#             advantages = returns - storage.v[i].detach()
#         else:
#             td_error = storage.reward[i] + config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
#             advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error
#         storage.advantage[i] = advantages.detach()
#         storage.ret[i] = returns.detach()
#
#     entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy'])
#     policy_loss = -(entries.log_pi_a * entries.advantage).mean()
#     value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
#     entropy_loss = entries.entropy.mean()
#
#     self.optimizer.zero_grad()
#     (policy_loss - config.entropy_weight * entropy_loss +
#      config.value_loss_weight * value_loss).backward()
#     nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
#     self.optimizer.step()
#
#     agent.switch_task()
#
# # Agent fns
# def save(self, filename):
#     torch.save(self.network.state_dict(), '%s.model' % (filename))
#     with open('%s.stats' % (filename), 'wb') as f:
#         pickle.dump(self.config.state_normalizer.state_dict(), f)
#
# def load(self, filename):
#     state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
#     self.network.load_state_dict(state_dict)
#     with open('%s.stats' % (filename), 'rb') as f:
#         self.config.state_normalizer.load_state_dict(pickle.load(f))
# def eval_step(self, state):
#     raise NotImplementedError
#
# def eval_episode(self):
#     env = self.config.eval_env
#     state = env.reset()
#     while True:
#         action = self.eval_step(state)
#         state, reward, done, info = env.step(action)
#         ret = info[0]['episodic_return']
#         if ret is not None:
#             break
#     return ret
#
# def eval_episodes(self):
#     episodic_returns = []
#     for ep in range(self.config.eval_episodes):
#         total_rewards = self.eval_episode()
#         episodic_returns.append(np.sum(total_rewards))
#     self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
#         self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
#     ))
#     self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
#     return {
#         'episodic_return_test': np.mean(episodic_returns),
#     }
#
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
#
# def switch_task(self):
#     config = self.config
#     if not config.tasks:
#         return
#     segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
#     if self.total_steps > segs[self.task_ind + 1]:
#         self.task_ind += 1
#         self.task = config.tasks[self.task_ind]
#         self.states = self.task.reset()
#         self.states = config.state_normalizer(self.states)
#
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
