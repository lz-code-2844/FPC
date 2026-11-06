#!/bin/bash
# Example data preprocessing script for FlowPredictor-Corrector

# Configuration
RAW_DATA_PATH=null  # 原始点云 TXT 目录
OUTPUT_DIR=null  # 预处理数据保存目录

# null 为待填写占位；填写后再运行，避免将空参数传入训练或推理。
for name in RAW_DATA_PATH OUTPUT_DIR; do
  if [[ "${!name}" == null ]]; then
    printf '请先填写参数：%s\n' "$name" >&2
    exit 2
  fi
done

echo "Starting data preprocessing..."
echo "=============================="

python generate_data_pkl.py \
    --input_path ${RAW_DATA_PATH} \
    --output_dir ${OUTPUT_DIR}

echo "Preprocessing completed!"
echo "Processed data saved to: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "  - ${OUTPUT_DIR}/train/"
echo "  - ${OUTPUT_DIR}/val/"
echo "  - ${OUTPUT_DIR}/train_data.pkl"
echo "  - ${OUTPUT_DIR}/val_data.pkl"
echo "  - ${OUTPUT_DIR}/train_out.csv"
echo "  - ${OUTPUT_DIR}/val_out.csv"
