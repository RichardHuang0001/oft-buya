import os
import torch
from diffusers import StableDiffusionPipeline

output_dir = "outputs/images/before_ft"
model_id = "jinaai/flat-2d-animerge"

# 这里的提示词必须和我们等会儿训练时用的保持完全一致！
prompt = "a 2d cartoon illustration of [V] duck, flat colors, black background"
negative_prompt = "3d, realistic, photorealistic, worst quality, low quality"

print(f"🚀 正在加载基座模型 {model_id} ...")
print("💡 第一次运行会自动下载模型权重，大概需要几分钟，请耐心等待。")

# 加载模型并发送到显卡 (使用 fp16 节省显存并提速)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

print("🎨 模型加载完毕，开始生成微调前的对比图...")

# 生成 3 张不同随机种子的图片作为报告素材
for i in range(3):
    generator = torch.Generator("cuda").manual_seed(42 + i)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    
    save_path = os.path.join(output_dir, f"before_ft_{i+1}.jpg")
    image.save(save_path)
    print(f"✅ 成功保存: {save_path}")

print("🎉 基线测试完成！")
