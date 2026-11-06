# FlowPredictor-Corrector Examples

This directory contains example scripts for using the FlowPredictor-Corrector (FPC) framework.

## Architecture Overview

FPC consists of two main components:
- **FlowPredictor** (`predictor.py`): Data-driven neural network
- **PhysicsCorrector** (`corrector.py`): Physics constraint module

## Available Examples

### 1. Data Preprocessing (`preprocess_example.sh`)

Preprocesses raw text data into the format required for training.

```bash
# Edit the paths in preprocess_example.sh, then run:
bash examples/preprocess_example.sh
```

### 2. Training (`train_example.sh`)

Trains the full FPC model (Predictor + Physics Corrector).

```bash
# Edit the paths and hyperparameters in train_example.sh, then run:
bash examples/train_example.sh
```

**Key Parameters:**
- `--rho`、`--mu`：血液密度（kg/m³）和恒定有效动力黏度（Pa·s），默认 `None`，由使用者填写。
- `--collocation_points`：每病例物理约束点数，默认 `None`。
- `--coordinate_scale`、`--field_scale`、`--field_offset`：当前病例网络输入/输出到 SI 单位的比例和偏移，默认 `None`；填写方式见 `corrector.py` 顶部中文说明。
- `--ema_alpha`：原梯度 EMA 平滑系数，按当前训练设置填写。
- `--no_physics`: Train predictor only (disable physics corrector)

### 3. Inference (`test_example.sh`)

Runs inference on test data using a trained FPC model.

```bash
# Edit the paths in test_example.sh, then run:
bash examples/test_example.sh
```

## Quick Start with Sample Data

To quickly test the model with the provided sample data:

```bash
# Run inference on the sample test case
bash examples/test_example.sh
```

## Python API Usage

### Using FPC Model Directly

```python
import torch
from fpc_model import FPCModel

# Create FPC model with physics correction
model = FPCModel(
    rho=None, mu=None, collocation_points=None,  # 填写血液密度、动力黏度和物理约束点数
    coordinate_scale=None, field_scale=None, field_offset=None,  # 填写当前病例的 SI 变换
    use_physics=True,
)
model = model.cuda()

# Load trained weights
model.load_state_dict(torch.load('path/to/weights/best_model.pth'))

# Inference
model.eval()
with torch.no_grad():
    predictions = model.predict(input_coords)
```

### Using Predictor Only

```python
from predictor import FlowPredictor

# Create predictor (data-driven only)
predictor = FlowPredictor(in_channels=4, out_channels=4)

# Forward pass
predictions = predictor(input_coords)
```

### Using Physics Corrector

```python
from corrector import PhysicsCorrector
from predictor import FlowPredictor

# Create components
predictor = FlowPredictor()
corrector = PhysicsCorrector(
    rho=None, mu=None, collocation_points=None,  # 计算前填写有效值
    coordinate_scale=None, field_scale=None, field_offset=None,  # 填写当前病例的 SI 变换
)

# Compute physics loss
physics_loss = corrector(predictor, coords)
```

## Customization

Each script contains configuration variables at the top that you should modify for your specific setup:

- **Data paths**: Update `*_INPUT`, `*_OUTPUT` paths to point to your data
- **Hyperparameters**: Adjust `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE` as needed
- **物理参数**：填写 `RHO`、`MU`、约束点数以及当前病例的 SI 比例和偏移；YAML/脚本中的待填值为 `null`，Python 中为 `None`。
- **Hardware**: Set `DEVICE` to `"cuda:0"` for GPU or `"cpu"` for CPU-only

## Notes

- Make the scripts executable: `chmod +x examples/*.sh`
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- GPU with sufficient VRAM is recommended for training (≥ 8GB recommended)
- The physics corrector requires gradients enabled during training
