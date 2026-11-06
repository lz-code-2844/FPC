"""
Training script for FlowPredictor-Corrector (FPC).

This script trains the FPC model using a combined data-driven and physics-based
loss function with adaptive gradient balancing.
"""

import os
import argparse
import pickle
import torch
import torch.optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from fpc_model import FPCModel
from dataset import pointdata

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def plot_training_history(dfhistory, loss_png_path):
    """Plot training and validation loss curves."""
    if 'train_mse' not in dfhistory.columns or 'val_mse' not in dfhistory.columns:
        print("Missing 'train_mse' or 'val_mse' column in history, unable to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Training MSE
    ax1.plot(dfhistory['epoch'], dfhistory['train_mse'], 'b-', label='Train MSE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train MSE', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_yscale('log')

    # Validation MSE
    ax2 = ax1.twinx()
    valid_val_mse = dfhistory[dfhistory['val_mse'] > 0]
    if not valid_val_mse.empty:
        ax2.plot(valid_val_mse['epoch'], valid_val_mse['val_mse'], 'r-', label='Val MSE')
        ax2.set_ylabel('Val MSE', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_yscale('log')

    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title('Training and Validation MSE')
    plt.savefig(loss_png_path)
    plt.close(fig)


def train_step(model, optimizer, coords, targets, ema_alpha):
    """Single training step."""
    model.train()
    optimizer.zero_grad()

    coords = coords.float().cuda()
    targets = targets.float().cuda()

    # Forward pass through predictor
    predictions = model(coords)

    # Compute combined loss (predictor + corrector)
    total_loss, loss_dict = model.compute_loss(
        predictions, targets, coords, ema_alpha=ema_alpha
    )

    # Backward pass
    total_loss.backward()
    optimizer.step()

    return loss_dict


def validate(model, dl_val):
    """Validation loop."""
    model.eval()
    val_loss_sum = 0.0
    val_mae_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for coords, targets in dl_val:
            coords = coords.float().cuda()
            targets = targets.float().cuda()

            predictions = model(coords)
            data_loss, mae = model.data_loss_fn(predictions, targets)

            val_loss_sum += data_loss.item()
            val_mae_sum += mae.item()
            val_count += 1

    return val_loss_sum / val_count, val_mae_sum / val_count


def train(model, dl_train, dl_val, epochs=100, start=1, lr=1e-3,
          steps_per_epoch=None, loss_csv=None, checkpoint_path=None,
          latest_model_dir=None, ema_alpha=0.99):
    """
    Main training loop for FPC.

    Args:
        model: FPCModel instance
        dl_train: Training data loader
        dl_val: Validation data loader
        epochs: Number of training epochs
        start: Starting epoch (for resuming)
        lr: Learning rate
        steps_per_epoch: Steps per epoch (None for full dataset)
        loss_csv: Path to save loss CSV
        checkpoint_path: Path to save/load checkpoint
        latest_model_dir: Directory to save model weights
        ema_alpha: EMA decay rate for gradient balancing
    """
    # Initialize history
    columns = ["epoch", "train_loss", "train_mae", "train_mse",
               "physics_loss", "lambda_physics",
               "val_loss", "val_mae", "val_mse"]
    dfhistory = pd.DataFrame(columns=columns)

    start_epoch = start
    min_loss = float('inf')

    # Setup model save path
    if latest_model_dir:
        os.makedirs(latest_model_dir, exist_ok=True)
        best_model_path = os.path.join(latest_model_dir, 'best_model.pth')
        print(f"Model weights will be saved to: {latest_model_dir}")
    else:
        default_dir = './outputs/weights'
        os.makedirs(default_dir, exist_ok=True)
        best_model_path = os.path.join(default_dir, 'best_model.pth')
        print(f"Warning: Using default save directory: {default_dir}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    # Load checkpoint if exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device('cuda:0'),
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        dfhistory = checkpoint['dfhistory']
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']
        model.grad_ema_data = checkpoint.get('grad_ema_data', 0.0)
        model.grad_ema_pinn = checkpoint.get('grad_ema_pinn', 0.0)
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")

    # Training loop
    for epoch in range(start_epoch, start + epochs):
        model.train()

        # Training metrics
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_mse_sum = 0.0
        physics_loss_sum = 0.0
        lambda_sum = 0.0
        step_count = 0

        for step, batch in enumerate(dl_train, 1):
            coords, targets = batch[0], batch[1]

            if steps_per_epoch and step > steps_per_epoch:
                break

            # Training step
            loss_dict = train_step(model, optimizer, coords, targets, ema_alpha)

            train_loss_sum += loss_dict['total_loss']
            train_mae_sum += loss_dict['mae']
            train_mse_sum += loss_dict['data_loss']
            physics_loss_sum += loss_dict['physics_loss']
            lambda_sum += loss_dict['lambda_physics']
            step_count += 1

            if step % 20 == 0:
                print(f"[Step {step}] MSE: {train_mse_sum/step_count:.5f}, "
                      f"MAE: {train_mae_sum/step_count:.3f}, "
                      f"Physics: {physics_loss_sum/step_count:.3f}, "
                      f"λ: {lambda_sum/step_count:.4f}")

        scheduler.step()
        step_count = max(step_count, 1)

        # Validation
        val_loss, val_mae = 0.0, 0.0
        if dl_val is not None:
            val_loss, val_mae = validate(model, dl_val)
            print(f"Validation at epoch {epoch}: Loss: {val_loss:.5f}, MAE: {val_mae:.5f}")

        # Record history
        info = {
            'epoch': epoch,
            'train_loss': train_loss_sum / step_count,
            'train_mae': train_mae_sum / step_count,
            'train_mse': train_mse_sum / step_count,
            'physics_loss': physics_loss_sum / step_count,
            'lambda_physics': lambda_sum / step_count,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'val_mse': val_loss
        }
        dfhistory = pd.concat([dfhistory, pd.DataFrame([info])], ignore_index=True)

        # Save loss CSV
        if loss_csv:
            dfhistory.to_csv(loss_csv, index=False)

        # Save checkpoint
        if checkpoint_path:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'dfhistory': dfhistory,
                'min_loss': min_loss,
                'grad_ema_data': model.grad_ema_data,
                'grad_ema_pinn': model.grad_ema_pinn
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")

        # Save best model
        current_loss = val_loss if dl_val is not None else (train_loss_sum / step_count)
        if current_loss < min_loss:
            min_loss = current_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model at epoch {epoch}, saved to {best_model_path}")

        # Print epoch summary
        print(f"Epoch {epoch}: Loss={info['train_loss']:.5f}, "
              f"MAE={info['train_mae']:.5f}, MSE={info['train_mse']:.5f}, "
              f"Physics={info['physics_loss']:.5f}, λ={info['lambda_physics']:.4f}")

        # Save periodic checkpoint
        if epoch % 20 == 0 and latest_model_dir:
            periodic_path = os.path.join(
                latest_model_dir, f'fpc_model_epoch_{epoch}.pth'
            )
            torch.save(model.state_dict(), periodic_path)

        # Update loss plot
        if loss_csv:
            loss_png_path = os.path.splitext(loss_csv)[0] + '.png'
            plot_training_history(dfhistory, loss_png_path)

    return dfhistory


def main():
    parser = argparse.ArgumentParser(description="FlowPredictor-Corrector Training")

    # Data paths
    parser.add_argument("--train_data_pkl", type=str, required=True,
                        help="Path to training dataset metadata (.pkl)")
    parser.add_argument("--val_data_pkl", type=str, default=None,
                        help="Path to validation dataset metadata (.pkl)")
    parser.add_argument("--train_input_path", type=str, required=True,
                        help="Directory containing training .txt files")
    parser.add_argument("--val_input_path", type=str, default=None,
                        help="Directory containing validation .txt files")
    parser.add_argument("--train_output_path", type=str, required=True,
                        help="CSV file with training file indices")
    parser.add_argument("--val_output_path", type=str, default=None,
                        help="CSV file with validation file indices")

    # Output paths
    parser.add_argument("--loss_csv", type=str, default='./outputs/logs/train_loss.csv',
                        help="CSV file to log training losses")
    parser.add_argument("--checkpoint", type=str, default='./outputs/checkpoints/checkpoint.pth',
                        help="Path to save/load checkpoint")
    parser.add_argument("--latest_model_dir", type=str, default='./outputs/weights',
                        help="Directory to save model weights")

    # Training settings
    parser.add_argument("--point_number", type=int, default=4096,
                        help="Number of points to sample per case")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Total number of training epochs")
    parser.add_argument("--step", type=int, default=None,
                        help="Steps per epoch (default: full dataset)")
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="Starting epoch for resuming training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="Device to use (cuda:0 or cpu)")
    parser.add_argument("--ema_alpha", type=float, default=0.99,
                        help="EMA decay rate for gradient balancing")

    # Physics settings
    parser.add_argument("--rho", type=float, default=None,
                        help="血液密度，kg/m³；填写与当前 CFD 一致的值")
    parser.add_argument("--mu", type=float, default=None,
                        help="恒定有效动力黏度，Pa·s；按正文填写 CFD Carreau 的无限剪切黏度")
    parser.add_argument("--collocation_points", type=int, default=None,
                        help="每病例物理约束采样点数，正整数，不超过输入点数")
    parser.add_argument("--coordinate_scale", type=float, nargs=4, default=None,
                        help="t,x,y,z 从网络数值到 SI 的四个比例，单位 s,m,m,m；匹配当前病例")
    parser.add_argument("--field_scale", type=float, nargs=4, default=None,
                        help="u,v,w,p 从网络数值到 SI 的四个比例，匹配当前病例归一化范围")
    parser.add_argument("--field_offset", type=float, nargs=4, default=None,
                        help="u,v,w,p 从网络数值到 SI 的四个偏移，匹配当前病例归一化最小值")
    parser.add_argument("--no_physics", action='store_true',
                        help="Disable physics correction (train predictor only)")

    args = parser.parse_args()

    # Create data loaders
    print("Loading training data...")
    train_data_info = pickle.load(open(args.train_data_pkl, 'rb'))
    data_train = pointdata(
        args.train_input_path, args.train_output_path,
        train_data_info, args.point_number
    )
    dl_train = DataLoader(
        data_train, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )

    dl_val = None
    if args.val_data_pkl and args.val_input_path and args.val_output_path:
        print("Loading validation data...")
        val_data_info = pickle.load(open(args.val_data_pkl, 'rb'))
        data_val = pointdata(
            args.val_input_path, args.val_output_path,
            val_data_info, args.point_number
        )
        dl_val = DataLoader(
            data_val, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers
        )

    # Create model
    print(f"Creating FPC model (use_physics={not args.no_physics})...")
    model = FPCModel(
        rho=args.rho, mu=args.mu, collocation_points=args.collocation_points,
        coordinate_scale=args.coordinate_scale, field_scale=args.field_scale,
        field_offset=args.field_offset,
        use_physics=not args.no_physics
    ).to(torch.device(args.device))

    # Train
    print(f"Starting training for {args.epochs} epochs...")
    history = train(
        model=model,
        dl_train=dl_train,
        dl_val=dl_val,
        epochs=args.epochs,
        start=args.start_epoch,
        lr=args.lr,
        steps_per_epoch=args.step,
        loss_csv=args.loss_csv,
        checkpoint_path=args.checkpoint,
        latest_model_dir=args.latest_model_dir,
        ema_alpha=args.ema_alpha
    )

    print("Training complete!")
    print(f"Final results saved to: {args.loss_csv}")


if __name__ == "__main__":
    main()
