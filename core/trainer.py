import os
import torch
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

from .executor import ExecutionContext, HookedExecutor, Phase
from .hooks import (
    Hook, OptimizerHook, LRSchedulerHook, LoggingHook,
    CheckpointHook, MetricHook
)
from utils import ConfigurationManager, LoggerFactory, CheckpointManager, MetricAggregator


class FPCTrainer(HookedExecutor):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ConfigurationManager.get_config()
        self.logger = LoggerFactory.get_logger(
            'FPCTrainer',
            'training.log'
        )
        self.metric_aggregator = MetricAggregator()
        self.checkpoint_manager = CheckpointManager(
            self.cfg.get('output.checkpoint_dir'),
            max_keep=self.cfg.get('training.checkpointing.max_keep', 5)
        )
        self._build_components()
        self._register_default_hooks()

    def _build_components(self):
        from fpc_model import FPCModel
        from torch.utils.data import DataLoader
        from dataset import pointdata
        import pickle

        self.device = torch.device(self.cfg.get('system.device', 'cuda:0'))

        self.model = FPCModel(
            rho=self.cfg.get('model.corrector.rho'),
            mu=self.cfg.get('model.corrector.mu'),
            collocation_points=self.cfg.get('model.corrector.collocation_points'),
            coordinate_scale=self.cfg.get('model.corrector.coordinate_scale'),
            field_scale=self.cfg.get('model.corrector.field_scale'),
            field_offset=self.cfg.get('model.corrector.field_offset'),
            use_physics=self.cfg.get('model.corrector.enabled', True)
        ).to(self.device)

        train_info = pickle.load(open(self.cfg.get('data.train_data_pkl'), 'rb'))
        self.train_dataset = pointdata(
            self.cfg.get('data.train_input_path'),
            self.cfg.get('data.train_output_path'),
            train_info,
            self.cfg.get('data.point_number', 4096)
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.get('training.batch_size', 4),
            shuffle=True,
            num_workers=self.cfg.get('system.num_workers', 4)
        )

        val_pkl = self.cfg.get('data.val_data_pkl')
        if val_pkl and os.path.exists(val_pkl):
            val_info = pickle.load(open(val_pkl, 'rb'))
            self.val_dataset = pointdata(
                self.cfg.get('data.val_input_path'),
                self.cfg.get('data.val_output_path'),
                val_info,
                self.cfg.get('data.point_number', 4096)
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.get('training.batch_size', 4),
                shuffle=False,
                num_workers=self.cfg.get('system.num_workers', 4)
            )
        else:
            self.val_loader = None

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.get('training.optimizer.lr', 1e-4),
            weight_decay=self.cfg.get('training.optimizer.weight_decay', 1e-5),
            betas=tuple(self.cfg.get('training.optimizer.betas', [0.9, 0.999]))
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.get('training.scheduler.T_max', 2000),
            eta_min=self.cfg.get('training.scheduler.eta_min', 1e-5)
        )

        self.start_epoch = 1
        self.best_loss = float('inf')

    def _register_default_hooks(self):
        self.register_hook(Phase.BEFORE_BACKWARD, OptimizerHook())
        self.register_hook(Phase.AFTER_BACKWARD, OptimizerHook())
        self.register_hook(Phase.EPOCH_END, LRSchedulerHook())
        self.register_hook(Phase.ITER_END, LoggingHook(
            interval=self.cfg.get('output.log_frequency', 20),
            logger=self.logger
        ))
        self.register_hook(Phase.EPOCH_END, MetricHook())
        self.register_hook(Phase.EPOCH_END, CheckpointHook(
            self.cfg.get('output.checkpoint_dir'),
            interval=self.cfg.get('training.checkpointing.frequency', 20)
        ))

    def _execute_iteration(self, batch):
        coords, targets = batch[0].to(self.device), batch[1].to(self.device)
        coords = coords.float()
        targets = targets.float()

        self.model.train()
        predictions = self.model(coords)

        ema_alpha = self.cfg.get('training.loss_balancing.ema_alpha', 0.99)
        loss, loss_dict = self.model.compute_loss(predictions, targets, coords, ema_alpha)

        self._context.loss = loss
        self._context.metrics = loss_dict

        self.execute_hooks(Phase.BEFORE_BACKWARD)
        self.execute_hooks(Phase.AFTER_BACKWARD)

        self.metric_aggregator.update(**loss_dict)

    def train(self, resume_from: Optional[str] = None):
        if resume_from:
            self._load_checkpoint(resume_from)

        context = ExecutionContext(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            data_loader=self.train_loader
        )

        self.logger.info(f"Starting training for {self.cfg.get('training.epochs')} epochs")
        self.run(context, self.train_loader, self.cfg.get('training.epochs'))
        self.logger.info("Training completed")

        self._save_best_model()

    def _load_checkpoint(self, path: str):
        self.logger.info(f"Loading checkpoint from {path}")
        state = self.checkpoint_manager.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.start_epoch = state['epoch'] + 1

    def _save_best_model(self):
        best_path = Path(self.cfg.get('output.weight_dir')) / 'best_model.pth'
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), best_path)
        self.logger.info(f"Best model saved to {best_path}")
