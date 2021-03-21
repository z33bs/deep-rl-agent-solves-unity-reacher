# Todo
# then trash storage which is unnecessary
# then look if can use clipping function for loss? Probably not necessary as diverges
# modularize in my own way

from config import Config
import time
import os
from a2c_agent import A2CAgent
from pathlib import Path
import numpy as np
import torch
from unity_environment import Env
import sys
# mkdir('log')
Path('log').mkdir(parents=True, exist_ok=True)
Path('data').mkdir(parents=True, exist_ok=True)

# set_one_thread()
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# torch.set_num_threads(1)

# seed
np.random.seed(333)
torch.manual_seed(np.random.randint(int(1e6)))

config = Config()
agent = A2CAgent(config, Env('reacher.app', is_mock=config.is_mock))
agent_name = agent.__class__.__name__
t0 = time.time()
episode = 1

avg_scores = []
eval_scores = []
num_eval_episodes = 100
target_avg_score = 30.0
agent_last_steps = 0

# Todo
# importable log for plotting

try:
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%s-%d' % (t0, agent_name, config.tag, agent.total_steps))
        if config.log_interval and not agent.total_steps % config.log_interval:
            sys.stdout.write('\r steps %d, running reward: %.4f, %.2f steps/s' % (
                agent.total_steps, np.array(agent.running_rewards).mean(), config.log_interval / (time.time() - t0)))
            # print('steps %d, running reward: %d, %.2f steps/s' % (
            #     agent.total_steps, np.array(agent.running_rewards).mean(), config.log_interval / (time.time() - t0))
            #       , end='\r')
            t0 = time.time()
        if config.max_episodes and episode - 1 >= config.max_episodes:
            if len(eval_scores) > 0:
                agent.save('data/%s-final-%.2f' % (config.tag, eval_scores[-1]))
            break

        is_done = agent.step()
        if is_done:
            # record avg reward for episode
            last_mean_reward = np.mean(agent.scores)
            avg_scores.append(last_mean_reward)

            # record rolling average for window of episodes
            eval_average = np.mean(np.array(avg_scores[-num_eval_episodes:])) \
                if len(avg_scores) > num_eval_episodes else np.mean(np.array(avg_scores))
            eval_std = np.std(np.array(avg_scores[-num_eval_episodes:])) \
                if len(avg_scores) > num_eval_episodes else np.std(np.array(avg_scores))
            eval_scores.append(eval_average)

            # log
            sys.stdout.write('\n')
            agent.logger.info('episode: %d, avg return: %.2f, eval return: %.2f & std eval return: %.2f - steps: %d' % (
                episode, last_mean_reward, eval_average, eval_std, agent.total_steps - agent_last_steps))

            agent.save('data/%s-%d-%.2f' % (config.tag, agent.total_steps, eval_average))

            agent_last_steps = agent.total_steps
            episode += 1
            agent.reset()

            if eval_average > target_avg_score:  # reached target
                break
finally:
    if len(eval_scores) > 0:
        agent.save('data/%s-stopped-%.2f' % (config.tag, eval_scores[-1]))
