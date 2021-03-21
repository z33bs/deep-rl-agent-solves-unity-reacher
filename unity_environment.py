from unityagents import UnityEnvironment
import numpy as np


class Env:
    def __init__(self, file_name, is_mock=False):
        self.is_mock = is_mock
        if is_mock:
            self.state = np.random.rand(20, 33)
            self.num_agents = 20
            self.action_size = 4
            self.state_size = 33
        else:
            self.env = UnityEnvironment(file_name)
            self.brain_name = self.env.brain_names[0]
            self.brain = self.env.brains[self.brain_name]
            self.env_info = self.env.reset(train_mode=False)[self.brain_name]

            self.state = self.env_info.vector_observations

            self.num_agents = len(self.env_info.agents)
            self.action_size = self.brain.vector_action_space_size
            self.state_size = self.state.shape[1]

    def reset(self, train_mode=True):
        if not self.is_mock:
            self.env_info = self.env.reset(train_mode)[self.brain_name]

        return self.state, self.num_agents, (self.state_size, self.action_size)

    def transition(self, action):
        if self.is_mock:
            return np.random.rand(self.num_agents).tolist(), \
                   np.random.rand(self.num_agents, self.state_size), \
                   list(map(bool, np.random.randint(0, 2, self.num_agents).tolist()))
            # list 20, ndarray 20,33, list 20 bool
        else:
            self.env_info = self.env.step(action)[self.brain_name]

            self.state = self.env_info.vector_observations  # get next state (for each agent)
            reward = self.env_info.rewards  # get reward (for each agent)
            done = self.env_info.local_done

            return reward, self.state, done
