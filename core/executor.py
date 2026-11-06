import torch
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field


class Phase(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    EPOCH_START = auto()
    EPOCH_END = auto()
    ITER_START = auto()
    ITER_END = auto()
    BEFORE_BACKWARD = auto()
    AFTER_BACKWARD = auto()


@dataclass
class ExecutionContext:
    epoch: int = 0
    max_epochs: int = 0
    iteration: int = 0
    max_iterations: int = 0
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[Any] = None
    data_loader: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    loss: Optional[torch.Tensor] = None
    phase: Phase = Phase.TRAIN

    def clear_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def step_grad(self):
        if self.optimizer is not None:
            self.optimizer.step()


class HookedExecutor:
    def __init__(self):
        self._hooks: Dict[Phase, List[Callable]] = {phase: [] for phase in Phase}
        self._context = ExecutionContext()

    def register_hook(self, phase: Phase, hook: Callable, priority: int = 0):
        self._hooks[phase].append((priority, hook))
        self._hooks[phase].sort(key=lambda x: x[0])

    def execute_hooks(self, phase: Phase, **kwargs):
        self._context.phase = phase
        for priority, hook in self._hooks[phase]:
            hook(self._context, **kwargs)

    def run(self, context: ExecutionContext, data_loader, max_epochs: int):
        self._context = context
        self._context.max_epochs = max_epochs

        for epoch in range(1, max_epochs + 1):
            self._context.epoch = epoch
            self.execute_hooks(Phase.EPOCH_START)

            for iteration, batch in enumerate(data_loader):
                self._context.iteration = iteration
                self.execute_hooks(Phase.ITER_START)

                self._execute_iteration(batch)

                self.execute_hooks(Phase.ITER_END)

            self.execute_hooks(Phase.EPOCH_END)

    def _execute_iteration(self, batch):
        raise NotImplementedError
