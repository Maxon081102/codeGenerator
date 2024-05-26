import torch

from typing import Union, Optional

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

class LRScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, lr: Union[list[int], int]) -> None:
        self.optimizer = optimizer
        # super().__init__(optimizer)
        self.lr = lr
        self.last_lr = [self.lr]
    
    def step(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_last_lr(self):
        raise NotImplementedError
    
    @staticmethod
    def set_lr(optimizer: Optimizer, lr: Union[list[int], int]):
        if isinstance(lr, list):
            assert len(lr) == len(optimizer.param_groups)
            for i, group in enumerate(optimizer.param_groups):
                group["lr"] = lr[i]
        else:
            for i, group in enumerate(optimizer.param_groups):
                group["lr"] = lr
    
class BitnetLRScheduler(LRScheduler):
    def __init__(
        self, 
        optimizer: Optimizer,
        num_warmup_steps: int, 
        num_training_steps: int,
        second_weight_decay: float,
        second_lr: float,
        start_lr: float =1e-10,
     ) -> None:
        assert isinstance(num_training_steps, int)
        assert isinstance(num_warmup_steps, int)
        
        super().__init__(optimizer, start_lr)
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.second_weight_decay = second_weight_decay
        self.half = int(num_training_steps / 2)
        assert self.half > num_warmup_steps
        
        warmup_rate = optimizer.param_groups[0]["lr"] - start_lr
        self.warump_rate = warmup_rate / num_warmup_steps
        
        self.second_warmup_rate = (second_lr - start_lr) / num_warmup_steps
        
        self.lr_rate = optimizer.param_groups[0]["lr"] / (num_training_steps - num_warmup_steps)
        self.second_lr_rate = second_lr / (num_training_steps - num_warmup_steps)
        
        self.second_lr = self.lr
        self.steps = 0
    
    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        self.last_lr[0] = self.lr
        if self.steps < self.num_warmup_steps:
            self.second_lr += self.second_warmup_rate
            self.lr += self.warump_rate
            self.set_lr(self.optimizer, self.lr)
        else:
            self.second_lr -= self.second_lr_rate
            self.lr -= self.lr_rate
            self.set_lr(self.optimizer, self.lr)
        
        if self.steps == self.half:
            self.lr, self.second_lr = self.second_lr, self.lr
            self.lr_rate, self.second_lr_rate = self.second_lr_rate, self.lr_rate
            for group in self.optimizer.param_groups:
                group["weight_decay"] = self.second_weight_decay
        self.steps += 1
        return self.lr

    def get_last_lr(self):
        return self.last_lr
        