import torch


class Config:
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    NOISY_LAYER_STD = 0.1

    def __init__(self):
        # note one episode lasts 100 000 steps
        self.is_mock = False
        self.hidden_dim = (256, 128)
        self.discount = 0.99
        self.use_gae = True
        self.gae_tau = 0.9
        self.entropy_weight = 0.01
        self.rollout_length = 5
        self.gradient_clip = 0.2
        self.log_level = 0
        self.tag = 'adam_256_128_clip0.2'
        self.value_loss_weight = 1.0
        self.log_interval = 20  # int(1e3)
        self.save_interval = None
        self.max_episodes = 10000
        self.rollout_min_reward = 1
        self.use_rollout_reward_threshold = True

        self.rollout_min_reward_start = 0.0001
        self.rollout_min_reward_max = 0.0005
        rollout_min_reward_incr_episodes = 50
        self.rollout_min_reward_incr = (self.rollout_min_reward_max - self.rollout_min_reward_start) \
                                       / (rollout_min_reward_incr_episodes - 1)
