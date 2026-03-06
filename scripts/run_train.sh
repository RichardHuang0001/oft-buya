#!/bin/bash
# 1. 强制清除环境中残留的死代理，防止网络被拦截
unset http_proxy https_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY

# 2. 强制使用 Hugging Face 国内镜像站直连
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4

echo "🚀 开始基于 flat-2d-animerge 训练真正的 OFT 不鸭模型 (运行在 GPU 4 上)..."

accelerate launch src/train_dreambooth_boft.py \
  --pretrained_model_name_or_path="jinaai/flat-2d-animerge" \
  --instance_data_dir="data/processed" \
  --output_dir="outputs/models" \
  --instance_prompt="a 2d cartoon illustration of [V] duck, flat colors, black background" \
  --validation_prompt="a 2d cartoon illustration of [V] duck, flat colors, black background" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --checkpointing_steps=800 \
  --use_8bit_adam \
  --use_boft \
  --report_to="tensorboard" \
  --wandb_project_name="oft_buya_project" \
  --wandb_run_name="oft_buya_run" \
  --seed="42"

echo "🎉 OFT 训练完成！权重已保存至 outputs/models/unet/800"
