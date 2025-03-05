from math import cos, pi
import torch


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps=2000, max_training_steps=40000, min_lr=1e-6):
        """
        optimizer: объект класса torch.optim.Optimizer
        warmup_steps: число шагов warmup
        max_training_steps: общее число шагов обучения
        min_lr: минимальное значение скорости обучения
        """

        # ваш код здесь
        self.optimizer = optimizer
        self.t_warmup = warmup_steps
        self.t_max = max_training_steps
        self.min_lr = min_lr
        self.max_lr = self.optimizer.param_groups[0]['lr']
        self.t = 1


    def step(self):
        """
        Делает шаг обновления скорости обучения optimizer.
        """

        # ваш код здесь
        if self.t <= self.t_warmup:
          lr = self.min_lr * (1 - self.t / self.t_warmup) + self.max_lr * (self.t / self.t_warmup)
        else:
          lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + cos(pi * (self.t - self.t_warmup) / (self.t_max - self.t_warmup)))
        self.t += 1
        #self.optimizer.param_groups[0]['lr'] = lr
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']