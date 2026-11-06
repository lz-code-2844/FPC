import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader

from utils import ConfigurationManager, LoggerFactory


class FPCInferenceEngine:
    def __init__(self, cfg=None):
        self.cfg = cfg or ConfigurationManager.get_config()
        self.logger = LoggerFactory.get_logger('Inference', 'inference.log')
        self.device = torch.device(self.cfg.get('system.device', 'cuda:0'))
        self._build_model()
        self._build_dataloader()

    def _build_model(self):
        from fpc_model import FPCModel
        self.model = FPCModel(
            use_physics=False
        ).to(self.device)
        self.model.eval()

    def _build_dataloader(self):
        from dataset_test import pointdata
        self.dataset = pointdata(
            self.cfg.get('data.test_input_path'),
            self.cfg.get('data.test_pkl_path'),
            self.cfg.get('data.point_number', 4096)
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.get('testing.batch_size', 1),
            shuffle=False
        )

    def run(self, checkpoint_path: str):
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        results = []
        for idx, (coords, targets) in enumerate(self.dataloader):
            result = self._infer_single(coords, targets)
            results.append(result)
            self.logger.info(f"Processed sample {idx + 1}/{len(self.dataloader)}")

        self._save_results(results)

    def _infer_single(self, coords: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        coords = coords.float().to(self.device)
        targets = targets.float().to(self.device)

        with torch.no_grad():
            predictions = self.model.predict(coords)

        mse = torch.nn.functional.mse_loss(predictions, targets).item()
        mae = torch.nn.functional.l1_loss(predictions, targets).item()

        return {
            'predictions': predictions.cpu().numpy(),
            'targets': targets.cpu().numpy(),
            'coords': coords.cpu().numpy(),
            'mse': mse,
            'mae': mae
        }

    def _save_results(self, results: List[Dict[str, Any]]):
        output_dir = Path(self.cfg.get('output.result_dir'))
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, result in enumerate(results):
            df = pd.DataFrame({
                'mse': [result['mse']],
                'mae': [result['mae']]
            })
            df.to_excel(output_dir / f'result_{idx}.xlsx', index=False)

        self.logger.info(f"Results saved to {output_dir}")
