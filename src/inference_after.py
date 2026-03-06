import os
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

output_dir = "outputs/images/after_ft"
os.makedirs(output_dir, exist_ok=True)
model_id = "jinaai/flat-2d-animerge"

# 因为我们在训练时默认设置了 500 步保存一次，所以我们去加载 checkpoint 500 的权重
adapter_dir = "outputs/models/unet/800/oft_buya_run" 

prompt = "a 2d cartoon illustration of [V] duck, flat colors, black background"
negative_prompt = "3d, realistic, photorealistic, worst quality, low quality"

print(f"🚀 正在加载基座模型 {model_id} ...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

print(f"🧩 正在融合我们刚刚训练好的 OFT 权重 ({adapter_dir}) ...")
# 将微调的 OFT 权重注入到 UNet 中
pipe.unet = PeftModel.from_pretrained(pipe.unet, adapter_dir)

print("🎨 模型融合完毕，开始生成微调后的【不鸭】专属图...")

for i in range(3):
    generator = torch.Generator("cuda").manual_seed(42 + i)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    
    save_path = os.path.join(output_dir, f"after_ft_{i+1}.jpg")
    image.save(save_path)
    print(f"✅ 成功保存专属不鸭图: {save_path}")

print("🎉 盲盒开启完毕！快去验收成果吧！")
