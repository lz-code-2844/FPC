#!/usr/bin/env python3
"""
FlowPredictor-Corrector Unified Entry Point
============================================

A professional, highly-encapsulated execution interface for FPC.

Usage:
    python run.py --config configs/default.yaml --mode train
    python run.py --config configs/default.yaml --mode test --checkpoint ./outputs/weights/best_model.pth
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def _initialize_environment():
    """Initialize system environment and CUDA settings."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONWARNINGS'] = 'ignore'

    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except ImportError:
        pass


def _parse_command_line():
    """Parse command line arguments with nested subparsers."""
    parser = argparse.ArgumentParser(
        prog='FPC',
        description='FlowPredictor-Corrector: Physics-Informed Point-Cloud Neural Network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Train:  python run.py --config configs/default.yaml --mode train
  Test:   python run.py --config configs/default.yaml --mode test --checkpoint best_model.pth
        '''
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['train', 'test', 'infer', 'preprocess'],
        required=True,
        help='Execution mode'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint path for resuming training or inference'
    )

    parser.add_argument(
        '--override',
        type=str,
        nargs='+',
        default=[],
        help='Override config values (e.g., training.epochs=100)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    return parser.parse_args()


def _apply_overrides(cfg, overrides):
    """Apply command-line overrides to configuration."""
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}. Use key=value")
        key, value = override.split('=', 1)

        # Try to convert value to appropriate type
        try:
            value = eval(value)
        except (NameError, SyntaxError):
            pass  # Keep as string

        cfg.set(key, value)


def _execute_training_mode(args, cfg):
    """Execute training workflow."""
    from core import FPCTrainer
    from utils import LoggerFactory

    LoggerFactory.initialize(cfg.get('output.log_dir'))
    logger = LoggerFactory.get_logger('Main', 'main.log')

    logger.info("=" * 60)
    logger.info("FlowPredictor-Corrector Training Pipeline")
    logger.info("=" * 60)

    trainer = FPCTrainer(cfg)
    trainer.train(resume_from=args.checkpoint)


def _execute_testing_mode(args, cfg):
    """Execute testing workflow."""
    from core.inference import FPCInferenceEngine
    from utils import LoggerFactory

    LoggerFactory.initialize(cfg.get('output.log_dir'))
    logger = LoggerFactory.get_logger('Main', 'main.log')

    logger.info("=" * 60)
    logger.info("FlowPredictor-Corrector Inference Engine")
    logger.info("=" * 60)

    engine = FPCInferenceEngine(cfg)
    checkpoint = args.checkpoint or cfg.get('output.weight_dir') + '/best_model.pth'
    engine.run(checkpoint)


def _execute_preprocess_mode(args, cfg):
    """Execute data preprocessing workflow."""
    import subprocess
    import sys

    input_path = cfg.get('data.train_input_path')
    output_dir = Path(input_path).parent

    cmd = [
        sys.executable,
        'generate_data_pkl.py',
        '--input_path', str(input_path),
        '--output_dir', str(output_dir)
    ]

    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    """Main entry point with exception handling."""
    _initialize_environment()
    args = _parse_command_line()

    try:
        # Initialize configuration
        from utils import ConfigurationManager
        ConfigurationManager.initialize(args.config)
        cfg = ConfigurationManager.get_config()

        # Apply overrides
        if args.override:
            _apply_overrides(cfg, args.override)

        # Set random seed if specified
        if args.seed is not None:
            import torch
            import numpy as np
            import random
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

        # Execute based on mode
        if args.mode == 'train':
            _execute_training_mode(args, cfg)
        elif args.mode == 'test':
            _execute_testing_mode(args, cfg)
        elif args.mode == 'infer':
            _execute_testing_mode(args, cfg)
        elif args.mode == 'preprocess':
            _execute_preprocess_mode(args, cfg)

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
