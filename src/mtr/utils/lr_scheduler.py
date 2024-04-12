import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupLinearLR(LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between
    warmup_start_lr and base_lr followed by a linear decay schedule to total_iters.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): steps for warmup to last
        total_steps (int): total steps that includes warmup and linear decay
        warmup_start_lr (float, optional): warmup start lr. Defaults to 0.0.
        start_factor (_type_, optional): start factor for linear decay. Defaults to 1/3.
        end_factor (float, optional): end factor for linear decay. Defaults to 1.0.
        last_epoch (int): The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        warmup_start_lr: float = 0.0,
        start_factor=1 / 3,
        end_factor=1.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start_lr = warmup_start_lr
        self.start_factor = start_factor
        self.end_factor = end_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Applies Linear LR decay with Linear warmup to the given optimizer params lr
        This function should be applied every step/batch
        Returns:
            list[float]: new learning rates
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )
        # warmup part
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)

        if self.last_epoch < self.warmup_steps:
            # apply linear warmup
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_steps - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        # if self.last_epoch == self.warmup_steps:
        #     return self.base_lrs

        # linear decay part
        if self.last_epoch == self.warmup_steps:
            return [lr * self.start_factor for lr in self.base_lrs]
            # return [group["lr"] * self.start_factor for group in self.optimizer.param_groups]

        if self.last_epoch > self.total_steps:
            # no change
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            group["lr"]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (self.total_steps * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor))
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        raise NotImplementedError
        return [
            base_lr
            * (
                self.start_factor
                + (self.end_factor - self.start_factor) * min(self.total_steps, self.last_epoch) / self.total_steps
            )
            for base_lr in self.base_lrs
        ]


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from torch import nn
    from torch.optim import AdamW

    model = nn.Sequential(nn.Linear(10, 2))
    optimizer = AdamW(model.parameters(), lr=0.001)

    warmup_steps = 500
    total_steps = 10000
    scheduler = WarmupLinearLR(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps, start_factor=1.0, end_factor=0
    )

    lrs = []
    for _ in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # print(lrs)
    plt.plot(lrs)
    plt.show()
