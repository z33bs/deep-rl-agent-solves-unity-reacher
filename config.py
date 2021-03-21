import torch


class Config:
    DEVICE = torch.device('cpu')
    NOISY_LAYER_STD = 0.1

    def __init__(self):
        # note one episode lasts 100 000 steps
        self.is_mock = True
        self.discount = 0.99
        self.use_gae = True
        self.gae_tau = 1.0
        self.entropy_weight = 0.01
        self.rollout_length = 5
        self.gradient_clip = 5
        self.log_level = 0
        self.tag = 'vanilla'
        self.value_loss_weight = 1.0
        self.log_interval = 10000  # int(1e3)
        self.save_interval = None
        self.max_episodes = 10
