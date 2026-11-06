import os
import time
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from .executor import ExecutionContext, Phase


class Hook(ABC):
    def __init__(self, priority: int = 0):
        self.priority = priority

    @abstractmethod
    def __call__(self, ctx: ExecutionContext, **kwargs):
        pass


class OptimizerHook(Hook):
    def __init__(self, grad_clip: Optional[float] = None):
        super().__init__(priority=0)
        self.grad_clip = grad_clip

    def __call__(self, ctx: ExecutionContext, **kwargs):
        if ctx.phase == Phase.BEFORE_BACKWARD:
            ctx.loss.backward()

        elif ctx.phase == Phase.AFTER_BACKWARD:
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    ctx.model.parameters(), self.grad_clip
                )
            ctx.step_grad()


class LRSchedulerHook(Hook):
    def __init__(self, step_frequency: str = "epoch"):
        super().__init__(priority=10)
        self.step_frequency = step_frequency

    def __call__(self, ctx: ExecutionContext, **kwargs):
        if self.step_frequency == "epoch" and ctx.phase == Phase.EPOCH_END:
            if ctx.scheduler is not None:
                ctx.scheduler.step()
        elif self.step_frequency == "iter" and ctx.phase == Phase.ITER_END:
            if ctx.scheduler is not None:
                ctx.scheduler.step()


class LoggingHook(Hook):
    def __init__(self, interval: int = 20, logger=None):
        super().__init__(priority=20)
        self.interval = interval
        self.logger = logger

    def __call__(self, ctx: ExecutionContext, **kwargs):
        if ctx.phase == Phase.ITER_END:
            if ctx.iteration % self.interval == 0:
                msg = f"Epoch [{ctx.epoch}/{ctx.max_epochs}] Iter [{ctx.iteration}] Loss: {ctx.loss.item():.6f}"
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)

        elif ctx.phase == Phase.EPOCH_END:
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in ctx.metrics.items()])
            msg = f"Epoch [{ctx.epoch}] completed. {metrics_str}"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)


class CheckpointHook(Hook):
    def __init__(self, checkpoint_dir: str, interval: int = 20, max_keep: int = 5):
        super().__init__(priority=30)
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval
        self.max_keep = max_keep
        self.checkpoints = []

    def __call__(self, ctx: ExecutionContext, **kwargs):
        if ctx.phase == Phase.EPOCH_END:
            if ctx.epoch % self.interval == 0 or ctx.epoch == ctx.max_epochs:
                self._save_checkpoint(ctx, f"epoch_{ctx.epoch}.pth")

    def _save_checkpoint(self, ctx: ExecutionContext, filename: str):
        state = {
            "epoch": ctx.epoch,
            "model_state_dict": ctx.model.state_dict(),
            "optimizer_state_dict": ctx.optimizer.state_dict() if ctx.optimizer else None,
            "scheduler_state_dict": ctx.scheduler.state_dict() if ctx.scheduler else None,
            "metrics": ctx.metrics,
        }

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, filepath)
        self.checkpoints.append(filepath)

        if len(self.checkpoints) > self.max_keep:
            old_ckpt = self.checkpoints.pop(0)
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)


class EvaluationHook(Hook):
    def __init__(self, val_loader, interval: int = 1):
        super().__init__(priority=15)
        self.val_loader = val_loader
        self.interval = interval

    def __call__(self, ctx: ExecutionContext, **kwargs):
        if ctx.phase == Phase.EPOCH_END:
            if ctx.epoch % self.interval == 0:
                self._evaluate(ctx)

    def _evaluate(self, ctx: ExecutionContext):
        ctx.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Evaluation logic
                count += 1

        ctx.metrics["val_loss"] = total_loss / max(count, 1)
        ctx.model.train()


class MetricHook(Hook):
    def __init__(self):
        super().__init__(priority=5)
        self.metric_history = []

    def __call__(self, ctx: ExecutionContext, **kwargs):
        if ctx.phase == Phase.EPOCH_END:
            self.metric_history.append({
                "epoch": ctx.epoch,
                **ctx.metrics
            })
