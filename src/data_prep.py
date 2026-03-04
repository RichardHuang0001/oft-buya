"""
数据预处理脚本

功能：
    1. 读取 data/raw/ 目录中的原始图片
    2. 将图片裁剪 / Resize 至 512×512（中心裁剪，保持宽高比）
    3. 统一重命名为 duck_XXXX.jpg 格式
    4. 输出至 data/processed/ 目录

使用方式：
    python src/data_prep.py
    
    或指定自定义路径：
    python src/data_prep.py --input_dir data/raw --output_dir data/processed --size 512

TODO:
    - [ ] 实现中心裁剪逻辑
    - [ ] 添加图片质量过滤（模糊图片排除）
    - [ ] 添加数据增强选项（水平翻转）
"""

# import os
# import argparse
# from pathlib import Path
# from PIL import Image
# from tqdm import tqdm

TARGET_SIZE = 512  # 目标图像边长（px）
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"


def center_crop_and_resize(image: "Image.Image", size: int) -> "Image.Image":
    """将图片中心裁剪为正方形并 Resize 到指定尺寸。"""
    # TODO: 实现中心裁剪
    pass


def process_images(input_dir: str, output_dir: str, size: int = TARGET_SIZE):
    """批量处理目录中的所有图片。"""
    # TODO: 实现批处理逻辑
    pass


if __name__ == "__main__":
    print(f"数据预处理脚本框架已就绪。")
    print(f"  输入目录：{INPUT_DIR}")
    print(f"  输出目录：{OUTPUT_DIR}")
    print(f"  目标尺寸：{TARGET_SIZE}×{TARGET_SIZE}")
    # process_images(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE)
