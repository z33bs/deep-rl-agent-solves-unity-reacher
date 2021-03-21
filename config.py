import torch


class Config:
    DEVICE = torch.device('cpu')
    NOISY_LAYER_STD = 0.1

    def __init__(self):
        self.discount = 0.99
        self.use_gae = True
        self.gae_tau = 1.0
        self.entropy_weight = 0.01
        self.rollout_length = 5
        self.gradient_clip = 5
        self.max_steps = int(2e7)
        self.log_level = 0
        self.tag = 'vanilla'
        self.value_loss_weight = 1.0
        # self.iteration_log_interval = 30
        self.log_interval = int(1e3)
        self.save_interval = int(1e5)
        self.eval_interval = int(1e3)
        self.eval_episodes = 1
