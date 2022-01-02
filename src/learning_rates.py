# coding=utf-8

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class AnnealingLR(_LRScheduler):
    """Anneals the learning rate from start to zero along a cosine curve."""

    DECAY_STYLES = ['linear', 'cosine', 'exponential', 'constant', 'None', 'noam']

    def __init__(self, optimizer, start_lr, warmup_iter, num_iters, decay_style=None, last_iter=-1, gradient_accumulation_steps=1):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = (warmup_iter // gradient_accumulation_steps) + 1
        self.num_iters = last_iter + 1
        self.end_iter = num_iters
        self.decay_style = decay_style.lower() if isinstance(decay_style, str) else None
        self.step(self.num_iters)
        if torch.distributed.get_rank() == 0:
            print('learning rate decaying', decay_style)

    def get_lr(self):
        # https://openreview.net/pdf?id=BJYwwY9ll pg. 4
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            if self.decay_style != self.DECAY_STYLES[5]:
                return float(self.start_lr) * self.num_iters / self.warmup_iter
            else:
                return float(self.start_lr) / math.sqrt(self.warmup_iter) * self.num_iters / self.warmup_iter #* self.num_iters / self.warmup_iter / math.sqrt(self.warmup_iter)
        else:
            if self.decay_style == self.DECAY_STYLES[0]:
                return self.start_lr*((self.end_iter-(self.num_iters-self.warmup_iter))/self.end_iter)
            elif self.decay_style == self.DECAY_STYLES[1]:
                return self.start_lr / 2.0 * (math.cos(math.pi * (self.num_iters - self.warmup_iter) / self.end_iter) + 1)
            elif self.decay_style == self.DECAY_STYLES[2]:
                #TODO: implement exponential decay
                return self.start_lr
            elif self.decay_style == self.DECAY_STYLES[5]:
                return self.start_lr / math.sqrt(self.num_iters)
            else:
                return self.start_lr

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

    def state_dict(self):
        sd = {
                'start_lr': self.start_lr,
                'warmup_iter': self.warmup_iter,
                'num_iters': self.num_iters,
                'decay_style': self.decay_style,
                'end_iter': self.end_iter
        }
        return sd

    def load_state_dict(self, sd):
        self.start_lr = sd['start_lr']
        self.warmup_iter = sd['warmup_iter']
        self.num_iters = sd['num_iters']
        self.end_iter = sd['end_iter']
        self.decay_style = sd['decay_style']
        self.step(self.num_iters)
