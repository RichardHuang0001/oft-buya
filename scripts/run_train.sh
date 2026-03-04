#!/usr/bin/env bash
# =============================================================================
#  run_train.sh — OFT/BOFT DreamBooth 一键训练脚本
#  
#  使用方式（在 4090 服务器上执行）：
#      bash scripts/run_train.sh
#
#  说明：
#      所有核心超参数集中在此文件中管理，修改超参数只需改这里，
#      无需动 src/train_dreambooth_boft.py 的核心逻辑。
# =============================================================================

set -e  # 遇到错误立即退出

# -------------------------
# 路径配置
# -------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_DATA_DIR="${PROJECT_ROOT}/data/processed"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/models/boft-run-01"
LOGGING_DIR="${PROJECT_ROOT}/outputs/logs/boft-run-01"

# -------------------------
# 模型配置
# -------------------------
PRETRAINED_MODEL="runwayml/stable-diffusion-v1-5"
INSTANCE_PROMPT="a photo of sks duck"

# -------------------------
# 训练超参数
# -------------------------
RESOLUTION=512
TRAIN_BATCH_SIZE=1
MAX_TRAIN_STEPS=800
LEARNING_RATE=1e-4
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0

# -------------------------
# OFT / BOFT 超参数
# -------------------------
BOFT_BLOCK_NUM=4
BOFT_BLOCK_SIZE=0
BOFT_N_BUTTERFLY_FACTOR=1
BOFT_DROPOUT=0.1

# -------------------------
# 日志与保存
# -------------------------
CHECKPOINTING_STEPS=200      # 每隔多少步保存一次 checkpoint
VALIDATION_STEPS=100         # 每隔多少步做一次验证推理

echo "================================================"
echo "  OFT-Buya 训练任务启动"
echo "  输出目录：${OUTPUT_DIR}"
echo "  最大训练步数：${MAX_TRAIN_STEPS}"
echo "  学习率：${LEARNING_RATE}"
echo "================================================"

# -------------------------
# 执行训练（待解注释）
# -------------------------
# accelerate launch "${PROJECT_ROOT}/src/train_dreambooth_boft.py" \
#     --pretrained_model_name_or_path="${PRETRAINED_MODEL}" \
#     --instance_data_dir="${INSTANCE_DATA_DIR}" \
#     --output_dir="${OUTPUT_DIR}" \
#     --logging_dir="${LOGGING_DIR}" \
#     --instance_prompt="${INSTANCE_PROMPT}" \
#     --resolution=${RESOLUTION} \
#     --train_batch_size=${TRAIN_BATCH_SIZE} \
#     --max_train_steps=${MAX_TRAIN_STEPS} \
#     --learning_rate=${LEARNING_RATE} \
#     --lr_scheduler="${LR_SCHEDULER}" \
#     --lr_warmup_steps=${LR_WARMUP_STEPS} \
#     --use_boft \
#     --boft_block_num=${BOFT_BLOCK_NUM} \
#     --boft_block_size=${BOFT_BLOCK_SIZE} \
#     --boft_n_butterfly_factor=${BOFT_N_BUTTERFLY_FACTOR} \
#     --boft_dropout=${BOFT_DROPOUT} \
#     --checkpointing_steps=${CHECKPOINTING_STEPS} \
#     --validation_steps=${VALIDATION_STEPS} \
#     --report_to="tensorboard"

echo "训练脚本框架已就绪，解注释上方命令后即可在 4090 上运行。"
