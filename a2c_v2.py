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
agent = A2CAgent(config)
agent_name = agent.__class__.__name__
t0 = time.time()
while True:
    if config.save_interval and not agent.total_steps % config.save_interval:
        agent.save('data/%s-%s-%s-%d' % (t0, agent_name, config.tag, agent.total_steps))
    if config.log_interval and not agent.total_steps % config.log_interval:
        agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
        t0 = time.time()
    if config.eval_interval and not agent.total_steps % config.eval_interval:
        agent.eval_episodes()
    if config.max_steps and agent.total_steps >= config.max_steps:
        agent.save('data/%s-%s-%s-%d' % (t0, agent_name, config.tag, agent.total_steps))
        break

    agent.step()
