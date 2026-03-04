import os
from PIL import Image

# 自动定位项目根目录，确保路径绝对安全
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_images():
    # 支持的图片格式
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    files = [f for f in os.listdir(RAW_DIR) if os.path.splitext(f)[1].lower() in valid_extensions]

    if not files:
        print(f"❌ 在 {RAW_DIR} 中没有找到图片，请检查路径。")
        return

    for i, filename in enumerate(files):
        img_path = os.path.join(RAW_DIR, filename)
        try:
            # 1. 打开图片并转为 RGBA（保留可能存在的透明通道）
            img = Image.open(img_path).convert("RGBA")

            # 2. 创建一个 512x512 的纯白底图 (画布)
            target_size = 512
            canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))

            # 3. 计算等比例放大/缩小的尺寸，让最长边变成 512
            img_ratio = img.width / img.height
            if img_ratio > 1:
                new_w = target_size
                new_h = int(target_size / img_ratio)
            else:
                new_h = target_size
                new_w = int(target_size * img_ratio)

            # 4. 缩放原图 (使用 LANCZOS 算法保证最高画质)
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # 5. 处理透明背景：将缩放后的图贴在纯白背景上，去掉透明度
            black_bg = Image.new("RGBA", img_resized.size, (0, 0, 0, 255))
            img_with_bg = Image.alpha_composite(black_bg, img_resized).convert("RGB")

            # 6. 计算居中粘贴的坐标
            paste_x = (target_size - new_w) // 2
            paste_y = (target_size - new_h) // 2

            # 7. 将处理好的图片贴到 512x512 画布的中央
            canvas.paste(img_with_bg, (paste_x, paste_y))

            # 8. 保存并规范重命名为 buya_01.jpg, buya_02.jpg...
            output_filename = f"buya_{i+1:02d}.jpg"
            output_path = os.path.join(PROCESSED_DIR, output_filename)
            canvas.save(output_path, "JPEG", quality=95)

            print(f"✅ 成功处理: {filename} -> {output_filename}")

        except Exception as e:
            print(f"❌ 处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    print("🚀 开始处理不鸭数据集...")
    process_images()
    print(f"🎉 处理完成！所有标准训练图已保存至 {PROCESSED_DIR}")