# OFT 不鸭微调项目 (oft-buya)

> 基于 **OFT (Orthogonal Fine-Tuning)** 对 Stable Diffusion 进行主题微调，实现"不鸭"风格的个性化图像生成。

---

## 📋 项目简介

本项目使用 [PEFT](https://github.com/huggingface/peft) 库中的 OFT / BOFT 方法，对 Stable Diffusion 基座模型进行参数高效微调（PEFT），使其能够生成特定主题（"不鸭"）的高质量图像。

**核心交付物：**
- ✅ 训练损失曲线 (Training Loss Curves)
- ✅ 微调前后对比图 (Before / After Fine-tuning)
- ✅ 项目报告 (3-page Report)

---

## 🗂️ 项目结构

```
oft-buya/
├── README.md                        # 项目说明文档
├── requirements.txt                 # Python 依赖列表
├── .gitignore                       # Git 忽略规则
│
├── data/
│   ├── raw/                         # 原始训练图片（不鸭截图/照片）
│   └── processed/                   # 预处理后的标准训练集 (512×512)
│
├── src/
│   ├── train_dreambooth_boft.py     # 核心训练脚本 (基于 HF 官方示例)
│   ├── data_prep.py                 # 数据预处理脚本（裁剪、重命名）
│   └── inference.py                 # 推理脚本（生成 Before/After 对比图）
│
├── scripts/
│   └── run_train.sh                 # 一键训练脚本（含超参数配置）
│
├── outputs/                         # 输出目录（已加入 .gitignore）
│   ├── models/                      # 训练保存的 OFT 权重
│   ├── logs/                        # TensorBoard 日志（用于 Loss 曲线）
│   └── images/
│       ├── before_ft/               # 微调前基座模型生成的图片
│       └── after_ft/                # 微调后生成的"不鸭"图片
│
└── report/
    ├── figures/                     # 报告插图（Loss 曲线截图、对比拼图）
    └── final_report.pdf             # 最终项目报告
```

---

## 🛠️ 环境安装

### 1. 克隆仓库

```bash
git clone https://github.com/<your-username>/oft-buya.git
cd oft-buya
```

### 2. 创建虚拟环境（推荐）

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 🚀 使用方法

### Step 1：数据预处理

将原始图片放入 `data/raw/`，然后运行：

```bash
python src/data_prep.py
```

处理后的 512×512 图片将保存至 `data/processed/`。

### Step 2：训练模型

在 4090 服务器上执行：

```bash
bash scripts/run_train.sh
```

训练日志将写入 `outputs/logs/`，可用 TensorBoard 查看：

```bash
tensorboard --logdir outputs/logs/
```

### Step 3：生成对比图

```bash
# 微调前（使用基座模型）
python src/inference.py --mode before

# 微调后（使用 OFT 权重）
python src/inference.py --mode after --lora_weights outputs/models/
```

生成的图片分别保存至 `outputs/images/before_ft/` 和 `outputs/images/after_ft/`。

---

## 📊 训练配置

| 参数 | 值 |
|------|-----|
| 基座模型 | `runwayml/stable-diffusion-v1-5` |
| 微调方法 | OFT / BOFT |
| 训练步数 | 800 |
| 学习率 | `1e-4` |
| 图像分辨率 | 512×512 |
| Batch Size | 1 |

> ⚠️ 具体超参数见 `scripts/run_train.sh`

---

## 📁 注意事项

- `outputs/models/` 中的权重文件**不会**上传至 GitHub（见 `.gitignore`）
- 权重文件请通过 Google Drive 或 HuggingFace Hub 共享
- 原始数据集同样不上传，请自行准备

---

## 📚 参考资料

- [PEFT 官方文档](https://huggingface.co/docs/peft)
- [DreamBooth 论文](https://arxiv.org/abs/2208.12242)
- [OFT 论文](https://arxiv.org/abs/2306.07280)
- [Diffusers DreamBooth 训练示例](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)
