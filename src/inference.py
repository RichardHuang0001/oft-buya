"""
推理脚本 —— 生成 Before / After 对比图

功能：
    - before 模式：使用基座模型（无微调）生成图片，保存至 outputs/images/before_ft/
    - after  模式：加载 OFT 权重后生成图片，保存至 outputs/images/after_ft/

使用方式：
    # 微调前（使用基座模型）
    python src/inference.py --mode before

    # 微调后（加载 OFT 权重）
    python src/inference.py --mode after --weights_path outputs/models/boft-run-01

TODO:
    - [ ] 实现 pipeline 加载与推理逻辑
    - [ ] 支持批量生成（--num_images 参数）
    - [ ] 添加 seed 固定，便于 Before/After 公平对比
    - [ ] 生成结果拼图（4宫格）并保存至 report/figures/
"""

# import torch
# import argparse
# from pathlib import Path
# from diffusers import StableDiffusionPipeline

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
PROMPT = "a photo of sks duck in a park, photorealistic"  # 推理使用的 prompt
NEGATIVE_PROMPT = "blurry, low quality, cartoon"
NUM_IMAGES = 4          # 每次生成图片数量
SEED = 42               # 随机种子（固定，便于复现对比）

OUTPUT_DIRS = {
    "before": "outputs/images/before_ft",
    "after":  "outputs/images/after_ft",
}


def load_pipeline(mode: str, weights_path: str = None):
    """加载推理 Pipeline。"""
    # TODO: 实现 pipeline 加载
    pass


def generate_images(pipeline, output_dir: str, num_images: int = NUM_IMAGES):
    """批量生成并保存图片。"""
    # TODO: 实现图片生成与保存
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Before / After 推理对比")
    parser.add_argument("--mode", choices=["before", "after"], required=True,
                        help="before: 基座模型 | after: 微调后模型")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="OFT 权重路径（--mode after 时需要指定）")
    args = parser.parse_args()

    print(f"推理模式：{args.mode}")
    print(f"输出目录：{OUTPUT_DIRS[args.mode]}")
    print("推理脚本框架已就绪，请填充具体推理逻辑。")
