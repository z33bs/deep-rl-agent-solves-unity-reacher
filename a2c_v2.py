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
episode = 0

avg_scores = []
eval_scores = []
num_eval_episodes = 100
target_avg_score = 30.0
agent_last_steps = 0

# Todo
# progress log avg score over x steps
# flexi n bootstrap
# if terminate, then no need for last step
# importable log for plotting

while True:
    episode += 1
    if config.save_interval and not agent.total_steps % config.save_interval:
        agent.save('data/%s-%s-%s-%d' % (t0, agent_name, config.tag, agent.total_steps))
    if config.log_interval and not agent.total_steps % config.log_interval:
        print('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
        t0 = time.time()
    if config.max_episodes and episode - 1 >= config.max_episodes:
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
        agent.logger.info(' episode: %d, avg return: %.2f, eval return: %.2f & std eval return: %.2f - steps: %d' % (
            episode, last_mean_reward, eval_average, eval_std, agent.total_steps - agent_last_steps))

        agent.save('data/%s-%d-%.2f' % (config.tag, agent.total_steps, eval_average))

        agent_last_steps = agent.total_steps
        agent.reset()
