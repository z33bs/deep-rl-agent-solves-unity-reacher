from unityagents import UnityEnvironment


class Env:
    def __init__(self, file_name):
        self.env = UnityEnvironment(file_name)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = self.env.reset(train_mode=False)[self.brain_name]

        self.state = self.env_info.vector_observations

        self.num_agents = len(self.env_info.agents)
        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.state.shape[1]

    def reset(self, train_mode=True):
        self.env_info = self.env.reset(train_mode)[self.brain_name]

        return self.state, self.num_agents, (self.state_size, self.action_size)

    def transition(self, action):
        self.env_info = self.env.step(action)[self.brain_name]

        self.state = self.env_info.vector_observations  # get next state (for each agent)
        reward = self.env_info.rewards  # get reward (for each agent)
        done = self.env_info.local_done

        return reward, self.state, done
