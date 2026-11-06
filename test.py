"""
Testing/Inference script for FlowPredictor-Corrector (FPC).

This script performs inference using a trained FPC model and saves
the predictions along with ground truth for evaluation.
"""

import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from fpc_model import FPCModel
from dataset_test import pointdata, read_ti


def get_minmax_from_raw(file_path):
    """Read raw txt file and return min/max for each column."""
    raw = read_ti(file_path)  # shape [N, 8], columns: t,x,y,z,u,v,w,p
    col_min = raw.min(axis=0)
    col_max = raw.max(axis=0)
    return col_min, col_max


def inv_minmax(arr_norm, vmin, vmax):
    """Denormalize array using min-max scaling."""
    return arr_norm * (vmax - vmin) + vmin


def save_results_to_excel(x, real_y, pred_y, raw_file, excel_file):
    """
    Denormalize and save prediction results to Excel.

    Args:
        x: Model input (1, 4, N) -> (t, x, y, z), normalized
        real_y: Ground truth values (N, 4), normalized
        pred_y: Predicted values (N, 4), normalized
        raw_file: Path to raw .txt file
        excel_file: Output Excel file path
    """
    col_min, col_max = get_minmax_from_raw(raw_file)

    # Process input coordinates
    x_np = np.squeeze(x, axis=0).T  # (N, 4)
    x_np[:, 1:4] = inv_minmax(x_np[:, 1:4], col_min[1:4], col_max[1:4])

    # Denormalize outputs
    real_denorm = inv_minmax(real_y, col_min[4:8], col_max[4:8])
    pred_denorm = inv_minmax(pred_y, col_min[4:8], col_max[4:8])

    # Create DataFrames
    df_x = pd.DataFrame(x_np, columns=['t', 'x', 'y', 'z'])
    df_real = pd.DataFrame(real_denorm, columns=['u_real', 'v_real', 'w_real', 'p_real'])
    df_pred = pd.DataFrame(pred_denorm, columns=['u_pred', 'v_pred', 'w_pred', 'p_pred'])

    # Merge and save
    merged_df = pd.concat([df_x, df_real, df_pred], axis=1)
    merged_df.to_excel(excel_file, index=False)
    print(f"Results saved to {excel_file}")


def test_model(input_path, pkl_path, weight_path, output_path,
               point_number=4096, device="cuda:0"):
    """
    Run inference with FPC model.

    Args:
        input_path: Path to input test data directory
        pkl_path: Path to test_data.pkl
        weight_path: Path to trained model weights
        output_path: Directory to save results
        point_number: Number of points to sample per file
        device: Device to run the model
    """
    # Create dataset and dataloader
    dataset = pointdata(input_path, pkl_path, point_number)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False
    )

    # Load model
    model = FPCModel().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    print(f"Loaded FPC model from {weight_path}")

    # Loss function for evaluation
    criterion = nn.MSELoss()

    # Inference loop
    all_mse = []
    all_mae = []

    for i, (x, y) in enumerate(dataloader):
        x = x.float().to(device)  # (1, 4, N)
        y = y.float().to(device)  # (1, 4, N)

        with torch.no_grad():
            pred = model.predict(x)  # Use predict method for inference

        # Convert to numpy
        real_y = y.squeeze(0).cpu().numpy().T  # (N, 4)
        pred_y = pred.squeeze(0).cpu().numpy().T  # (N, 4)
        x_np = x.cpu().numpy()

        # Compute metrics
        mse = criterion(torch.tensor(pred_y), torch.tensor(real_y)).item()
        mae = np.mean(np.abs(pred_y - real_y))

        all_mse.append(mse)
        all_mae.append(mae)

        print(f"Sample {i}: MSE={mse:.6f}, MAE={mae:.6f}")

        # Save results
        raw_file = os.path.join(input_path, f"{i}.txt")
        excel_file = os.path.join(output_path, f"result_{i}.xlsx")
        save_results_to_excel(x_np, real_y, pred_y, raw_file, excel_file)

    # Print overall metrics
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(all_mse)}")
    print(f"Mean MSE: {np.mean(all_mse):.6f} (+/- {np.std(all_mse):.6f})")
    print(f"Mean MAE: {np.mean(all_mae):.6f} (+/- {np.std(all_mae):.6f})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Test FlowPredictor-Corrector Model"
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input test data directory")
    parser.add_argument("--pkl_path", type=str, required=True,
                        help="Path to test_data.pkl")
    parser.add_argument("--weight_path", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--point_number", type=int, default=4096,
                        help="Number of points to sample per file")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run the model (cuda:0 or cpu)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Run test
    test_model(
        input_path=args.input_path,
        pkl_path=args.pkl_path,
        weight_path=args.weight_path,
        output_path=args.output_path,
        point_number=args.point_number,
        device=args.device
    )


if __name__ == "__main__":
    main()
