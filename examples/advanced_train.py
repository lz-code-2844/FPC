#!/usr/bin/env python3
"""
Advanced Training Example with Full Encapsulation
=================================================

This example demonstrates the professional, layered architecture of FPC.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    # Layer 1: Configuration Management (Singleton Pattern)
    from utils import ConfigurationManager
    ConfigurationManager.initialize('./configs/default.yaml')
    cfg = ConfigurationManager.get_config()

    print("=" * 60)
    print("FlowPredictor-Corrector Professional Architecture")
    print("=" * 60)
    print(f"\n[Layer 1] Configuration Loaded:")
    print(f"  - Training epochs: {cfg.get('training.epochs')}")
    print(f"  - Batch size: {cfg.get('training.batch_size')}")
    print(f"  - Device: {cfg.get('system.device')}")

    # Layer 2: Logger Factory (Factory Pattern)
    from utils import LoggerFactory
    LoggerFactory.initialize(cfg.get('output.log_dir'))
    logger = LoggerFactory.get_logger('Demo', 'demo.log')
    logger.info("Logger initialized")

    # Layer 3: Registry System (Registry Pattern)
    from utils import MODELS, LOSSES
    print(f"\n[Layer 2] Registry System:")
    print(f"  - Registered models: {list(MODELS.module_dict.keys())}")
    print(f"  - Registered losses: {list(LOSSES.module_dict.keys())}")

    # Layer 4: Core Execution Framework (Hook Pattern)
    from core import FPCTrainer, Phase
    print(f"\n[Layer 3] Execution Framework:")
    print(f"  - Available phases: {[p.name for p in Phase]}")

    # Layer 5: Trainer (Facade Pattern)
    trainer = FPCTrainer(cfg)
    print(f"\n[Layer 4] Trainer Initialized:")
    print(f"  - Model: {trainer.model.__class__.__name__}")
    print(f"  - Optimizer: {trainer.optimizer.__class__.__name__}")
    print(f"  - Scheduler: {trainer.scheduler.__class__.__name__}")

    print("\n" + "=" * 60)
    print("Architecture demonstration complete.")
    print("Run 'python run.py --config configs/default.yaml --mode train'")
    print("=" * 60)


if __name__ == '__main__':
    main()
