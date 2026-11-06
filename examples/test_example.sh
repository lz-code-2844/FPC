#!/bin/bash
# Example inference script for FlowPredictor-Corrector

# Configuration
TEST_INPUT=null  # 测试点云 TXT 目录
TEST_PKL=null  # 测试元数据 pkl 文件路径
WEIGHT_PATH=null  # 用于预测的权重文件路径
OUTPUT_PATH=null  # 预测结果保存目录
POINT_NUMBER=null  # 每病例采样点数，正整数
DEVICE=null  # 设备标识，填写 cpu 或 cuda:设备编号

# null 为待填写占位；填写后再运行，避免将空参数传入训练或推理。
for name in TEST_INPUT TEST_PKL WEIGHT_PATH OUTPUT_PATH POINT_NUMBER DEVICE; do
  if [[ "${!name}" == null ]]; then
    printf '请先填写参数：%s\n' "$name" >&2
    exit 2
  fi
done

echo "=========================================="
echo "FlowPredictor-Corrector Inference"
echo "=========================================="

python test.py \
    --input_path ${TEST_INPUT} \
    --pkl_path ${TEST_PKL} \
    --weight_path ${WEIGHT_PATH} \
    --output_path ${OUTPUT_PATH} \
    --point_number ${POINT_NUMBER} \
    --device ${DEVICE}

echo "=========================================="
echo "Inference completed!"
echo "Results saved to: ${OUTPUT_PATH}/"
echo "=========================================="
