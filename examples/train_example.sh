#!/bin/bash
# Example training script for FlowPredictor-Corrector

# Configuration
TRAIN_DATA_PKL=null  # 训练元数据 pkl 文件路径
VAL_DATA_PKL=null  # 验证元数据 pkl 文件路径
TRAIN_INPUT=null  # 训练点云 TXT 目录
VAL_INPUT=null  # 验证点云 TXT 目录
TRAIN_OUTPUT=null  # 训练病例索引 CSV 文件路径
VAL_OUTPUT=null  # 验证病例索引 CSV 文件路径
CHECKPOINT=null  # checkpoint 保存或恢复路径
WEIGHT_DIR=null  # 模型权重保存目录
LOSS_CSV=null  # 训练日志 CSV 文件路径

# Training hyperparameters
POINT_NUMBER=null  # 每病例采样点数，正整数
BATCH_SIZE=null  # 训练批大小，正整数
EPOCHS=null  # 训练轮数，正整数
LEARNING_RATE=null  # 学习率，正数
DEVICE=null  # 设备标识，填写 cpu 或 cuda:设备编号
NUM_WORKERS=null  # 数据加载进程数，非负整数
EMA_ALPHA=null  # 原梯度 EMA 平滑系数，无量纲

# Physics corrector settings
RHO=null  # 血液密度，kg/m³
MU=null  # 恒定有效动力黏度，Pa·s；按正文填写 CFD Carreau 的无限剪切黏度
COLLOCATION_POINTS=null  # 每病例物理约束点数，正整数
COORDINATE_SCALE=null  # t,x,y,z 转为 SI 的四个比例，空格分隔；匹配当前病例
FIELD_SCALE=null  # u,v,w,p 转为 SI 的四个比例，空格分隔；匹配当前病例
FIELD_OFFSET=null  # u,v,w,p 转为 SI 的四个偏移，空格分隔；匹配当前病例

# null 为待填写占位；填写后再运行，避免将空参数传入训练或推理。
for name in TRAIN_DATA_PKL VAL_DATA_PKL TRAIN_INPUT VAL_INPUT TRAIN_OUTPUT VAL_OUTPUT CHECKPOINT WEIGHT_DIR LOSS_CSV POINT_NUMBER BATCH_SIZE EPOCHS LEARNING_RATE DEVICE NUM_WORKERS EMA_ALPHA RHO MU COLLOCATION_POINTS COORDINATE_SCALE FIELD_SCALE FIELD_OFFSET; do
  if [[ "${!name}" == null ]]; then
    printf '请先填写参数：%s\n' "$name" >&2
    exit 2
  fi
done

echo "=========================================="
echo "FlowPredictor-Corrector Training"
echo "=========================================="
echo "Predictor: PointNet++ based neural network"
echo "Corrector: Navier-Stokes physics constraints"
echo "=========================================="

python train.py \
    --train_data_pkl ${TRAIN_DATA_PKL} \
    --val_data_pkl ${VAL_DATA_PKL} \
    --train_input_path ${TRAIN_INPUT} \
    --val_input_path ${VAL_INPUT} \
    --train_output_path ${TRAIN_OUTPUT} \
    --val_output_path ${VAL_OUTPUT} \
    --checkpoint ${CHECKPOINT} \
    --latest_model_dir ${WEIGHT_DIR} \
    --loss_csv ${LOSS_CSV} \
    --point_number ${POINT_NUMBER} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --num_workers ${NUM_WORKERS} \
    --device ${DEVICE} \
    --ema_alpha ${EMA_ALPHA} \
    --rho ${RHO} \
    --mu ${MU} \
    --collocation_points ${COLLOCATION_POINTS} \
    --coordinate_scale ${COORDINATE_SCALE} \
    --field_scale ${FIELD_SCALE} \
    --field_offset ${FIELD_OFFSET}

echo "=========================================="
echo "Training completed!"
echo "Best model: ${WEIGHT_DIR}/best_model.pth"
echo "Training logs: ${LOSS_CSV}"
echo "=========================================="
