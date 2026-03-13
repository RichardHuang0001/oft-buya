<div align="right">
  <a href="./README.md"><img src="https://img.shields.io/badge/lang-English-blue?style=flat-square" /></a>
  <a href="./README_zh.md"><img src="https://img.shields.io/badge/语言-中文-red?style=flat-square" /></a>
</div>

# oft-buya

基于 [BOFT](https://arxiv.org/abs/2311.06243)（蝶形正交微调）+ DreamBooth，对 Stable Diffusion 进行主题微调，使其能生成"不鸭"角色风格的图像。

---

## 项目结构

```
oft-buya/
├── data/
│   ├── raw/                         # 原始训练图片
│   └── processed/                   # 预处理后的 512×512 训练集
├── src/
│   ├── train_dreambooth_boft.py     # 训练脚本
│   ├── data_prep.py                 # 数据预处理（裁剪、缩放）
│   ├── inference_after.py           # 使用微调权重推理
│   └── eval_clip_score.py           # CLIP 分数评估
├── scripts/
│   └── run_train.sh                 # 一键训练启动脚本
├── outputs/                         # 已加入 .gitignore
│   ├── models/                      # 保存的 BOFT adapter 权重
│   ├── logs/                        # TensorBoard 日志
│   └── images/
│       ├── before_ft/               # 微调前生成图
│       └── after_ft/                # 微调后生成图
└── report/
    ├── figures/
    └── final_report.pdf
```

---

## 安装

```bash
git clone https://github.com/RichardHuang0001/oft-buya.git
cd oft-buya
pip install -r requirements.txt
```

---

## 使用流程

**1. 数据预处理**

```bash
python src/data_prep.py
```

将 `data/raw/` 中的图片裁剪并缩放为 512×512，输出至 `data/processed/`。

**2. 训练**

```bash
bash scripts/run_train.sh
```

用 TensorBoard 查看训练过程：

```bash
tensorboard --logdir outputs/models/logs/
```

**3. 推理**

```bash
python src/inference_after.py
```

**4. 评估（CLIP 分数）**

```bash
python src/eval_clip_score.py
```

---

## 训练配置

| 参数 | 值 |
|------|----|
| 基座模型 | `jinaai/flat-2d-animerge` |
| 微调方法 | BOFT（via PEFT） |
| Instance Prompt | `a 2d cartoon illustration of sks duck, flat colors, black background` |
| 训练步数 | 800 |
| 学习率 | `2e-4` |
| 图像分辨率 | 512×512 |
| Batch Size | 2 |

完整超参数见 `scripts/run_train.sh`。

---

## 参考资料

- [PEFT](https://github.com/huggingface/peft)
- [BOFT 论文](https://arxiv.org/abs/2311.06243)
- [DreamBooth 论文](https://arxiv.org/abs/2208.12242)
- [Diffusers DreamBooth 示例](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)
