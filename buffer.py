#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'mask'])
PrioritizedTransition = namedtuple('Transition',
                                   ['state', 'action', 'reward', 'next_state', 'mask', 'sampling_prob', 'idx'])


class Storage:
    def __init__(self, memory_size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['state', 'action', 'reward', 'mask',
                       'v', 'q', 'pi', 'log_pi', 'entropy',
                       'advantage', 'ret', 'q_a', 'log_pi_a',
                       'mean', 'next_state']
        self.keys = keys
        self.memory_size = memory_size
        self.reset()

    def feed(self, data):
        for k, v in data.items():
            if k not in self.keys:
                raise RuntimeError('Undefined key')
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.memory_size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])
        self.pos = 0
        self._size = 0

    def extract(self, keys):
        data = [getattr(self, k)[:self.memory_size] for k in keys]
        data = map(lambda x: torch.cat(x, dim=0), data)
        Entry = namedtuple('Entry', keys)
        return Entry(*list(data))

