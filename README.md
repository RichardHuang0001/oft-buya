# oft-buya

Fine-tuning Stable Diffusion on the "Buya" duck character using [BOFT](https://arxiv.org/abs/2311.06243) (Butterfly Orthogonal Fine-Tuning) via DreamBooth.

---

## Project Structure

```
oft-buya/
├── data/
│   ├── raw/                         # original images
│   └── processed/                   # preprocessed 512×512 training set
├── src/
│   ├── train_dreambooth_boft.py     # training script
│   ├── data_prep.py                 # preprocessing: crop & resize
│   ├── inference_after.py           # inference with fine-tuned weights
│   └── eval_clip_score.py           # CLIP score evaluation
├── scripts/
│   └── run_train.sh                 # training launcher
├── outputs/                         # ignored by git
│   ├── models/                      # saved BOFT adapter weights
│   ├── logs/                        # TensorBoard logs
│   └── images/
│       ├── before_ft/
│       └── after_ft/
└── report/
    ├── figures/
    └── final_report.pdf
```

---

## Setup

```bash
git clone https://github.com/RichardHuang0001/oft-buya.git
cd oft-buya
pip install -r requirements.txt
```

---

## Usage

**1. Preprocess data**

```bash
python src/data_prep.py
```

Resizes images in `data/raw/` to 512×512 and saves them to `data/processed/`.

**2. Train**

```bash
bash scripts/run_train.sh
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir outputs/models/logs/
```

**3. Inference**

```bash
python src/inference_after.py
```

**4. Evaluate (CLIP score)**

```bash
python src/eval_clip_score.py
```

---

## Training Config

| Parameter | Value |
|-----------|-------|
| Base model | `jinaai/flat-2d-animerge` |
| Method | BOFT (via PEFT) |
| Instance prompt | `a 2d cartoon illustration of sks duck, flat colors, black background` |
| Steps | 800 |
| Learning rate | `2e-4` |
| Resolution | 512×512 |
| Batch size | 2 |

See `scripts/run_train.sh` for the full config.

---

## References

- [PEFT](https://github.com/huggingface/peft)
- [BOFT paper](https://arxiv.org/abs/2311.06243)
- [DreamBooth paper](https://arxiv.org/abs/2208.12242)
- [Diffusers DreamBooth example](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)
