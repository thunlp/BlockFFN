import math
import bmtrain as bmt


class Cosine(bmt.lr_scheduler.WarmupLRScheduler):
    r"""
    After a warmup period during which learning rate increases linearly between 0 and the start_lr,
    The decay period performs :math:`\text{lr}=\text{start_lr}\times \dfrac{1+\cos \left( \pi \cdot \dfrac{\text{num_iter}-\text{warmup_iter}}{\text{end_iter}-\text{warmup_iter}}\right)}{2}`
    """

    def __init__(
        self, optimizer, start_lr, warmup_iter, end_iter, num_iter=0, lr_end_restart=0, resume_no_optimze=0
    ) -> None:
        self.warmup_iter = warmup_iter
        self.end_iter = end_iter
        self.optimizer = optimizer
        self.num_iter = num_iter
        self._current_lr = None
        self._start_lr = start_lr
        self.start_lr = []
        self.lr_end_restart = lr_end_restart
        self.resume_step = num_iter
        self.resume_no_optimze = resume_no_optimze
        for group in self.optimizer.param_groups:
            self.start_lr.append(group["lr"])

        self.step(self.num_iter)

    def get_lr_warmup(self, num_iter, base_lr) -> float:
        return base_lr * num_iter / self.warmup_iter

    def get_lr_decay(self, num_iter, base_lr) -> float:
        progress = (num_iter - self.warmup_iter) / max(1, (self.end_iter - self.warmup_iter))
        if progress > 1:
            if self.lr_end_restart == 0:
                progress = 1
            elif self.lr_end_restart == 1:
                progress = progress
            elif self.lr_end_restart == 2:
                progress = int(progress) * 2 + (progress - 1)

        return max(base_lr * 0.1, base_lr * (0.1 + 0.45 * (1.0 + math.cos(progress * math.pi))))

    def get_lr(self, base_lr):
        assert self.num_iter >= 0
        if self.resume_step + self.resume_no_optimze > self.num_iter:
            bmt.print_rank("resume no optimize")
            return 0

        if self.num_iter < self.warmup_iter:
            return self.get_lr_warmup(self.num_iter, base_lr)
        else:
            return self.get_lr_decay(self.num_iter, base_lr)

    @property
    def current_lr(self):
        return self._current_lr

    def step(self, num_iter=None) -> None:
        if num_iter is None:
            num_iter = self.num_iter + 1
        self.num_iter = num_iter

        self._current_lr = self.get_lr(self._start_lr)
        for group, base_lr in zip(self.optimizer.param_groups, self.start_lr):
            group["lr"] = self.get_lr(base_lr)

    def state_dict(self):
        return {
            "_start_lr": self._start_lr,
            "start_lr": self.start_lr,
            "warmup_iter": self.warmup_iter,
            "end_iter": self.end_iter,
            "num_iter": self.num_iter,
        }

    def load_state_dict(self, state_dict):
        self._start_lr = state_dict["_start_lr"]
        self.start_lr = state_dict["start_lr"]
        self.warmup_iter = state_dict["warmup_iter"]
        self.end_iter = state_dict["end_iter"]
        self.num_iter = state_dict["num_iter"]

        self.step(self.num_iter)


class WarmupStableDrop(bmt.lr_scheduler.WarmupLRScheduler):
    r"""
    After a warmup period during which learning rate increases linearly between 0 and the start_lr,
    The decay period performs :math:`\text{lr}=\text{start_lr}\times \dfrac{1+\cos \left( \pi \cdot \dfrac{\text{num_iter}-\text{warmup_iter}}{\text{end_iter}-\text{warmup_iter}}\right)}{2}`
    """

    def __init__(
        self, optimizer, start_lr, warmup_iter, end_iter, drop_iter=0, num_iter=0, resume_no_optimze=0
    ) -> None:
        self.warmup_iter = warmup_iter
        self.end_iter = end_iter
        self.drop_iter = drop_iter
        self.optimizer = optimizer
        self.num_iter = num_iter
        self._current_lr = None
        self._start_lr = start_lr
        self.start_lr = []
        self.resume_step = num_iter
        self.resume_no_optimze = resume_no_optimze
        for group in self.optimizer.param_groups:
            self.start_lr.append(group["lr"])

        self.step(self.num_iter)

    def get_lr_warmup(self, num_iter, base_lr, warmup_iter) -> float:
        return base_lr * num_iter / warmup_iter

    def get_lr_stable(self, num_iter, base_lr):
        return base_lr

    def get_lr_drop(self, num_iter, base_lr):
        progress = (self.end_iter - num_iter) / self.drop_iter
        return base_lr * (0.1 + max(0.9 * (self.end_iter - num_iter) / self.drop_iter, 0))

    def get_lr(self, base_lr):
        assert self.num_iter >= 0
        if self.resume_step + self.resume_no_optimze > self.num_iter:
            return self.get_lr_warmup(self.num_iter - self.resume_step, base_lr, self.resume_no_optimze)

        if self.num_iter < self.warmup_iter:
            return self.get_lr_warmup(self.num_iter, base_lr, self.warmup_iter)

        if self.num_iter > self.end_iter - self.drop_iter:
            return self.get_lr_drop(self.num_iter, base_lr)

        return self.get_lr_stable(self.num_iter, base_lr)

    @property
    def current_lr(self):
        return self._current_lr

    def step(self, num_iter=None) -> None:
        if num_iter is None:
            num_iter = self.num_iter + 1
        self.num_iter = num_iter

        self._current_lr = self.get_lr(self._start_lr)
        for group, base_lr in zip(self.optimizer.param_groups, self.start_lr):
            group["lr"] = self.get_lr(base_lr)

    def state_dict(self):
        return {
            "_start_lr": self._start_lr,
            "start_lr": self.start_lr,
            "warmup_iter": self.warmup_iter,
            "end_iter": self.end_iter,
            "num_iter": self.num_iter,
        }

    def load_state_dict(self, state_dict):
        self._start_lr = state_dict["_start_lr"]
        self.start_lr = state_dict["start_lr"]
        self.warmup_iter = state_dict["warmup_iter"]
        self.end_iter = state_dict["end_iter"]
        self.num_iter = state_dict["num_iter"]

        self.step(self.num_iter)



class WarmupStableExp(bmt.lr_scheduler.WarmupLRScheduler):
    r"""
    After a warmup period during which learning rate increases linearly between 0 and the start_lr,
    The decay period performs :math:`\text{lr}=\text{start_lr}\times \dfrac{1+\cos \left( \pi \cdot \dfrac{\text{num_iter}-\text{warmup_iter}}{\text{end_iter}-\text{warmup_iter}}\right)}{2}`
    """

    def __init__(
        self, optimizer, start_lr, warmup_iter, drop_begin=-1, drop_rate=0.5, drop_iter=0, num_iter=0, resume_no_optimze=0
    ) -> None:
        
        self.warmup_iter = warmup_iter
        self.drop_iter = drop_iter
        self.optimizer = optimizer
        self.num_iter = num_iter
        self._current_lr = None
        self._start_lr = start_lr
        self.start_lr = []
        self.resume_step = num_iter
        self.resume_no_optimze = resume_no_optimze
        self.drop_begin = drop_begin
        self.drop_iter = drop_iter  # here drop_iter is half-life
        self.drop_rate = drop_rate
        for group in self.optimizer.param_groups:
            self.start_lr.append(group["lr"])

        self.step(self.num_iter)

    def get_lr_warmup(self, num_iter, base_lr, warmup_iter) -> float:
        return base_lr * num_iter / warmup_iter

    def get_lr_stable(self, num_iter, base_lr):
        return base_lr

    def get_lr_drop(self, num_iter, base_lr):
        if self.drop_iter < 0:
            return base_lr 
        progress = (num_iter - self.drop_begin) / self.drop_iter
        return base_lr * (self.drop_rate ** progress)

    def get_lr(self, base_lr):
        assert self.num_iter >= 0
        if self.resume_step + self.resume_no_optimze > self.num_iter:
            return self.get_lr_warmup(self.num_iter - self.resume_step, base_lr, self.resume_no_optimze)

        if self.num_iter < self.warmup_iter:
            return self.get_lr_warmup(self.num_iter, base_lr, self.warmup_iter)

        if self.num_iter > self.drop_begin:
            return self.get_lr_drop(self.num_iter, base_lr)

        return self.get_lr_stable(self.num_iter, base_lr)

    @property
    def current_lr(self):
        return self._current_lr

    def step(self, num_iter=None) -> None:
        if num_iter is None:
            num_iter = self.num_iter + 1
        self.num_iter = num_iter

        self._current_lr = self.get_lr(self._start_lr)
        for group, base_lr in zip(self.optimizer.param_groups, self.start_lr):
            group["lr"] = self.get_lr(base_lr)

    def state_dict(self):
        return {
            "_start_lr": self._start_lr,
            "start_lr": self.start_lr,
            "warmup_iter": self.warmup_iter,
            "drop_begin": self.drop_begin,
            "num_iter": self.num_iter,
        }

    def load_state_dict(self, state_dict):
        self._start_lr = state_dict["_start_lr"]
        self.start_lr = state_dict["start_lr"]
        self.warmup_iter = state_dict["warmup_iter"]
        self.drop_begin = state_dict["drop_begin"]
        self.num_iter = state_dict["num_iter"]

        self.step(self.num_iter)
