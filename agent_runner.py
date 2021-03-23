import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from ddpg_agent import Agent
from unity_environment import Env
import time
from pathlib import Path

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train(env, agent):
    log = Logger()
    scores = []
    scores_window = deque(maxlen=100)
    running_score = deque(maxlen=100)
    n_episodes = 1000
    is_solved_threshold = 30.0

    last_max_score = 0

    for episode in range(n_episodes):
        states, _, _ = env.reset()
        agent.reset()  # reset the agent noise
        score = np.zeros(env.num_agents)

        while True:
            actions = agent.act(states)

            rewards, next_states, dones = env.transition(actions)

            t, l = agent.step(states, actions, np.expand_dims(np.array(rewards), axis=1),
                              next_states,
                              np.expand_dims(np.array(dones), axis=1))

            score += rewards
            running_score.append(np.mean(rewards))
            states = next_states  # roll over the state to next time step

            print('Collection: {}   Learns: {}  Last 100 rewards: {}'.format(t, l, np.mean(running_score)), end='\r')

            if np.any(dones):  # exit loop if episode finished
                break

        avg_score = np.mean(score)
        scores.append(np.mean(score))
        scores_window.append(avg_score)
        if avg_score > last_max_score:
            agent.save()
            last_max_score = avg_score

        print('\n')
        log.write_line('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode + 1, np.mean(score),
                                                                                            np.mean(scores_window)))

        if np.mean(scores_window) >= is_solved_threshold and episode > 99:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode + 1,
                                                                                       np.mean(scores_window)))
            agent.save('data/final')
            break

    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    env.close()


def plot_scores(path):
    losses = np.genfromtxt(path)
    losses = losses[~np.isnan(losses)].reshape(-1,3)
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[1], label='Score', alpha=0.5)
    plt.plot(losses.T[2], label='100 episode average', alpha=0.5)
    plt.plot([30]*len(losses), label='Target')
    plt.title("Average Score over 20 agents")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.show()


def test(env, agent):
    for episode in range(3):

        score = np.zeros(env.num_agents)

        while True:
            actions = agent.act(env.state, add_noise=False)

            rewards, states, dones = env.transition(actions)
            score += rewards

            if np.any(dones):
                break

        print('Episode: \t{} \tScore: \t{:.2f}'.format(episode, np.mean(score)))

    env.close()


class Logger():
    def __init__(self, path='log/ddpg_'):
        Path('log').mkdir(parents=True, exist_ok=True)
        self.path = path + '{}'.format(time.time()) + '.txt'

    def write_line(self, log_txt, copy_to_screen=True):
        with open(self.path, 'a+') as log_file:
            log_file.write(log_txt + '\n')
        print(log_txt)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'plot'])
    parser.add_argument('--env', type=str, default='reacher.app')
    parser.add_argument('--mock', type=bool, default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--seed', type=int, default=round(time.time()))
    args = parser.parse_args()

    if args.mode == 'plot':
        print('Plotting scores...\n')
        plot_scores(args.load)
    else:

        print('Loading environment and instantiating agent...\n')
        env = Env(args.env, is_mock=args.mock)
        try:
            agent = Agent(env.state_size, env.action_size, env.num_agents, args.seed, args.load)

            if args.mode == 'test':
                print("Running in evaluation mode...\n")
                test(env, agent)
            else:
                print("Training agent...\n")
                try:
                    train(env, agent)
                finally:
                    agent.save()
        finally:
            env.close()
