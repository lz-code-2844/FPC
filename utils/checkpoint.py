import os
import torch
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
from collections import OrderedDict


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_keep: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.checkpoints = []
        self.best_score = None
        self.best_checkpoint = None

    def save(self, state: Dict[str, Any], filename: str, is_best: bool = False) -> str:
        filepath = self.checkpoint_dir / filename
        torch.save(state, filepath)
        self.checkpoints.append(filepath)

        if len(self.checkpoints) > self.max_keep:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists() and 'best' not in old_checkpoint.name:
                old_checkpoint.unlink()

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            shutil.copy(filepath, best_path)
            self.best_checkpoint = best_path

        return str(filepath)

    def load(self, filename: Optional[str] = None, map_location: str = 'cpu') -> Dict[str, Any]:
        if filename is None:
            filename = self.best_checkpoint or self.checkpoints[-1]

        filepath = Path(filename)
        if not filepath.exists():
            filepath = self.checkpoint_dir / filepath

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        return torch.load(filepath, map_location=map_location)

    def get_latest(self) -> Optional[str]:
        return str(self.checkpoints[-1]) if self.checkpoints else None

    def get_best(self) -> Optional[str]:
        return str(self.best_checkpoint) if self.best_checkpoint else None
