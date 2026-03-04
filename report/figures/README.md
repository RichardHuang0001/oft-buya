# 项目报告素材

本目录用于存放插入最终报告的所有图片素材：

| 文件名（建议） | 内容说明 |
|----------------|----------|
| `loss_curve.png` | TensorBoard 截图导出的训练损失曲线 |
| `before_after_grid.png` | 微调前后 4×2 对比拼图 |
| `qualitative_results.png` | 多样化生成结果展示 |

## 生成 Loss 曲线截图步骤

1. 在任意机器上安装 TensorBoard：`pip install tensorboard`
2. 启动 TensorBoard：`tensorboard --logdir outputs/logs/`
3. 打开浏览器访问 `http://localhost:6006`
4. 在 **Scalars** 面板找到 `train/loss`，截图保存至本目录
