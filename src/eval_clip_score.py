import torch
import os
import glob
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F


# ──────────────────────────────────────────────
# ★ 配置区：按需修改以下路径和 prompt
# ──────────────────────────────────────────────
REF_IMAGE_DIR   = "./data/processed"          # 15 张参考图所在目录
AFTER_IMAGE_DIR = "./outputs/images/after_ft" # 微调后生成图目录（3张）
BEFORE_IMAGE_DIR= "./outputs/images/before_ft"# 微调前基线图目录（3张，可选）
PROMPT          = "[V] duck in anime style"   # 生成时使用的 prompt
MODEL_ID        = "openai/clip-vit-base-patch32"
# ──────────────────────────────────────────────


def load_images_from_dir(dir_path: str, exts=("*.jpg", "*.png", "*.jpeg")):
    """按文件名升序加载目录下所有图片，返回 (PIL.Image列表, 文件名列表)。"""
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(dir_path, ext)))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"[!] 目录 '{dir_path}' 下未找到图片文件。")
    images = [Image.open(p).convert("RGB") for p in paths]
    names  = [os.path.basename(p) for p in paths]
    return images, names

@torch.no_grad()
def encode_images(model, processor, images, device):
    """批量编码图片，返回 L2 归一化后的特征矩阵 (N, D)。"""
    inputs = processor(images=images, return_tensors="pt").to(device)
    feats  = model.get_image_features(**inputs)
    return F.normalize(feats, p=2, dim=-1)


@torch.no_grad()
def encode_text(model, processor, text: str, device):
    """编码单条文本，返回 L2 归一化后的特征向量 (1, D)。"""
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    feats  = model.get_text_features(**inputs)
    return F.normalize(feats, p=2, dim=-1)


def eval_group(label: str, gen_images, gen_names, ref_embeds, text_embed, model, processor, device):
    """
    对一组生成图计算两项指标，逐张打印，并返回均值。

    - Prompt Fidelity  : 每张生成图与 prompt 文本的余弦相似度，取均值
    - Subject Fidelity : 每张生成图与『所有参考图』余弦相似度的均值，再对所有生成图取均值
                         （即 mean over generated images of mean over reference images）
    """
    print(f"\n{'─'*50}")
    print(f"  [{label}] 共 {len(gen_images)} 张生成图")
    print(f"{'─'*50}")

    gen_embeds = encode_images(model, processor, gen_images, device)  # (N_gen, D)

    pf_scores = []  # Prompt Fidelity per image
    sf_scores = []  # Subject Fidelity per image

    for i, name in enumerate(gen_names):
        g = gen_embeds[i:i+1]  # (1, D)

        # Prompt Fidelity：生成图 vs. 文本
        pf = torch.matmul(g, text_embed.T).item()

        # Subject Fidelity：生成图 vs. 全部参考图，取均值
        # ref_embeds: (N_ref, D)  →  sim: (1, N_ref)
        sf = torch.matmul(g, ref_embeds.T).mean().item()

        pf_scores.append(pf)
        sf_scores.append(sf)
        print(f"    {name:25s}  PF={pf:.4f}   SF={sf:.4f}")

    mean_pf = sum(pf_scores) / len(pf_scores)
    mean_sf = sum(sf_scores) / len(sf_scores)
    print(f"  {'平均 (mean)':23s}  PF={mean_pf:.4f}   SF={mean_sf:.4f}")
    return mean_pf, mean_sf


def calculate_clip_scores():
    # 1. 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 正在使用设备: {device}")

    # 2. 加载 CLIP 模型
    print(f"[*] 正在加载 CLIP 模型 ({MODEL_ID})...")
    try:
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        model     = CLIPModel.from_pretrained(MODEL_ID).to(device)
        model.eval()
    except Exception as e:
        print(f"[!] 模型加载失败: {e}")
        return

    # 3. 加载参考图（训练集，15张）
    print(f"[*] 加载参考图（训练集）from: {REF_IMAGE_DIR}")
    try:
        ref_images, ref_names = load_images_from_dir(REF_IMAGE_DIR)
    except FileNotFoundError as e:
        print(e); return
    print(f"    共加载 {len(ref_images)} 张参考图: {ref_names}")

    # 4. 预计算参考图特征（只算一次，复用）
    ref_embeds  = encode_images(model, processor, ref_images, device)   # (15, D)

    # 5. 预计算文本特征
    text_embed  = encode_text(model, processor, PROMPT, device)          # (1, D)
    print(f"[*] Prompt: \"{PROMPT}\"")

    # 6. 评估各组生成图
    results = {}

    # ── 微调后（after_ft）
    print(f"\n[*] 加载微调后生成图 from: {AFTER_IMAGE_DIR}")
    try:
        after_images, after_names = load_images_from_dir(AFTER_IMAGE_DIR)
        results["After FT"] = eval_group(
            "After FT", after_images, after_names,
            ref_embeds, text_embed, model, processor, device
        )
    except FileNotFoundError as e:
        print(e)

    # ── 微调前基线（before_ft，可选）
    if os.path.isdir(BEFORE_IMAGE_DIR):
        print(f"\n[*] 加载微调前基线图 from: {BEFORE_IMAGE_DIR}")
        try:
            before_images, before_names = load_images_from_dir(BEFORE_IMAGE_DIR)
            results["Before FT"] = eval_group(
                "Before FT", before_images, before_names,
                ref_embeds, text_embed, model, processor, device
            )
        except FileNotFoundError as e:
            print(e)

    # 7. 汇总报告
    print("\n" + "="*56)
    print("📊 定量评测汇总 (Quantitative Evaluation Summary)")
    print("="*56)
    print(f"  {'组别':<14} {'Prompt Fidelity':>18} {'Subject Fidelity':>18}")
    print(f"  {'─'*12} {'─'*18} {'─'*18}")
    for label, (pf, sf) in results.items():
        print(f"  {label:<14} {pf:>18.4f} {sf:>18.4f}")
    print("="*56)
    print("💡 参考标准：")
    print("  - Prompt Fidelity  > 0.30 表示生成图符合 Prompt 语义。")
    print("  - Subject Fidelity > 0.70 表示生成角色与原始主体高度一致。")
    if len(results) == 2:
        delta_pf = results["After FT"][0] - results["Before FT"][0]
        delta_sf = results["After FT"][1] - results["Before FT"][1]
        print(f"\n  📈 微调增益 (After - Before):")
        print(f"     Prompt Fidelity  Δ = {delta_pf:+.4f}")
        print(f"     Subject Fidelity Δ = {delta_sf:+.4f}")
    print("="*56)


if __name__ == "__main__":
    calculate_clip_scores()