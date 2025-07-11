import math
import bmtrain as bmt
from abc import ABC
import numpy as np


class BaseValueScheduler(ABC):

    val_schedulers = []

    @classmethod
    def add_scheduler(cls, scheduler):
        cls.val_schedulers.append(scheduler)

    @classmethod
    def step_all(cls, num_iter=None) -> None:
        for scheduler in cls.val_schedulers:
            scheduler.step(num_iter)
        if len(cls.val_schedulers) > 0:
            scheduler = cls.val_schedulers[0]
            bmt.print_rank(f">>> step {scheduler.num_iter} with value {scheduler.current_val} <<<")

    def __init__(
        self, start_val, warmup_iter, end_iter, num_iter=0, target_ratio=0.,
    ) -> None:
        self.warmup_iter = warmup_iter
        self.end_iter = end_iter
        self.num_iter = num_iter
        self._current_val = None
        self._start_val = start_val
        self.resume_step = num_iter
        self.target_ratio = target_ratio

        self.step(self.num_iter)
        BaseValueScheduler.add_scheduler(self)

    def get_val_warmup(self, num_iter, base_val) -> float:
        return base_val * num_iter / self.warmup_iter

    @property
    def current_val(self):
        return self._current_val

    def get_val_cur(self, num_iter, base_val) -> float:
        pass

    def get_val(self, base_val):
        assert self.num_iter >= 0
        if self.resume_step > self.num_iter:
            bmt.print_rank("resume no optimize")
            return 0

        if self.num_iter < self.warmup_iter:
            return self.get_val_warmup(self.num_iter, base_val)
        else:
            return self.get_val_cur(self.num_iter, base_val)

    def step(self, num_iter=None) -> None:
        if num_iter is None:
            num_iter = self.num_iter + 1
        self.num_iter = num_iter

        self._current_val = self.get_val(self._start_val)

    def state_dict(self):
        return {
            "_start_val": self._start_val,
            "_current_val": self._current_val,
            "warmup_iter": self.warmup_iter,
            "end_iter": self.end_iter,
            "num_iter": self.num_iter,
            "resume_step": self.resume_step,
            "target_ratio": self.target_ratio,
        }

    def load_state_dict(self, state_dict):
        self._start_val = state_dict["_start_val"]
        self._current_val = state_dict["_current_val"]
        self.warmup_iter = state_dict["warmup_iter"]
        self.end_iter = state_dict["end_iter"]
        self.num_iter = state_dict["num_iter"]
        self.resume_step = state_dict["resume_step"]
        self.target_ratio = state_dict["target_ratio"]

        self.step(self.num_iter)


class CosineScheduler(BaseValueScheduler):
    def get_val_cur(self, num_iter, base_val) -> float:
        progress = (num_iter - self.warmup_iter) / max(1, (self.end_iter - self.warmup_iter))
        progress = min(1, progress)
        if self.target_ratio <= 1:
            return base_val * max(self.target_ratio, self.target_ratio + 0.5 * (1.0 - self.target_ratio) * (1.0 + math.cos(progress * math.pi)))
        else:
            return base_val * min(self.target_ratio, self.target_ratio + 0.5 * (1.0 - self.target_ratio) * (1.0 + math.cos(progress * math.pi)))


class LinearScheduler(BaseValueScheduler):
    def get_val_cur(self, num_iter, base_val):
        progress = (num_iter - self.warmup_iter) / max(1, (self.end_iter - self.warmup_iter))
        progress = 1 - min(1, progress)
        if self.target_ratio <= 1:
            return base_val * (self.target_ratio + max((1.0 - self.target_ratio) * progress, 0))
        else:
            return base_val * (self.target_ratio + min((1.0 - self.target_ratio) * progress, 0))


class ExpScheduler(BaseValueScheduler):
    """
    f(x) = base_val * (target_ratio ** progress)
    s.t. f(0) = base_val; f(1) = base_val * target_ratio
    """
    def get_val_cur(self, num_iter, base_val):
        progress = (num_iter - self.warmup_iter) / max(1, (self.end_iter - self.warmup_iter))
        progress = min(1, progress)
        return base_val * (self.target_ratio ** progress)


class AdaptiveLinearScheduler:
    def __init__(self, start_val, warmup_iter, stable_iter, step_iter, start_step: int = 0, max_factor: float = 1.0, min_factor=1.0):
        self.init_start_val = start_val
        self.cur_start_val = start_val
        self.warmup_iter = warmup_iter
        self.stable_iter = stable_iter
        self.step_iter = step_iter
        self.start_step = start_step
        self.max_factor = max_factor
        self.min_factor = min_factor
        assert warmup_iter <= stable_iter

        self.min_loss_container, self.max_loss_container = [], []
        self.cur_scheduler = LinearScheduler(
            self.cur_start_val, warmup_iter=0, end_iter=self.stable_iter, num_iter=0, target_ratio=1.0,
        )

    def step(self, iteration: int, loss: float, can_step: bool = False):
        iteration -= self.start_step
        assert iteration > 0, f"{iteration} | {self.start_step}"
        # record current loss value
        self.max_loss_container.append(loss)
        min_loss = np.mean(self.min_loss_container) if len(self.min_loss_container) > 0 else -1
        max_loss = np.mean(self.max_loss_container) if len(self.max_loss_container) > 0 else -1
        if iteration % self.step_iter == 0:
            self.min_loss_container = self.max_loss_container
            self.max_loss_container = []

        if iteration < self.warmup_iter:
            self.cur_scheduler.step()
            res_val = self.init_start_val * (iteration / self.warmup_iter)
            return res_val
        # update and record scheduler
        res_val = self.cur_scheduler.current_val
        if can_step and iteration >= self.stable_iter and iteration % self.step_iter == 0:
            self.cur_start_val = res_val
            assert min_loss > 0 and max_loss > 0
            target_ratio = max_loss / min_loss
            base_target_ratio = max(self.min_factor, target_ratio) if target_ratio > 1 else target_ratio
            target_ratio = min(self.init_start_val * self.max_factor / self.cur_start_val, base_target_ratio)
            self.cur_scheduler = LinearScheduler(
                self.cur_start_val, warmup_iter=0, end_iter=self.step_iter, num_iter=0, target_ratio=target_ratio,
            )
            bmt.print_rank(
                "*" * 10, f"AdaptiveLinearScheduler: change to LinearScheduler from {self.cur_start_val:.6f}, "
                          f"{self.step_iter} steps, min {min_loss:.4f} / max {max_loss:.4f} / target_ratio {target_ratio:.4f}", "*" * 10)
        self.cur_scheduler.step()
        return res_val

    def state_dict(self):
        return {
            "init_start_val": self.init_start_val,
            "cur_start_val": self.cur_start_val,
            "warmup_iter": self.warmup_iter,
            "stable_iter": self.stable_iter,
            "step_iter": self.step_iter,
            "start_step": self.start_step,
            "max_factor": self.max_factor,
            "min_factor": self.min_factor,
            "min_loss_container": self.min_loss_container,
            "max_loss_container": self.max_loss_container,
            "cur_scheduler": self.cur_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.init_start_val = state_dict["init_start_val"]
        self.cur_start_val = state_dict["cur_start_val"]
        self.warmup_iter = state_dict["warmup_iter"]
        self.stable_iter = state_dict["stable_iter"]
        self.step_iter = state_dict["step_iter"]
        self.start_step = state_dict["start_step"]
        self.max_factor = state_dict["max_factor"]
        self.min_factor = state_dict["min_factor"]
        self.min_loss_container = state_dict["min_loss_container"]
        self.max_loss_container = state_dict["max_loss_container"]
        self.cur_scheduler = LinearScheduler(0, 0, 0)
        self.cur_scheduler.load_state_dict(state_dict["cur_scheduler"])
