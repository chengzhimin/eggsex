# eggsex — 鸡胚图像与高光谱科研建模

**v0.2.0：可训练深度模型 + 系统分组评估。** 研究 E3.5–E4.0 鸡胚的图像、光谱及联合输入是否包含可复现的性别预测信息。本仓库尚未用真实鸡胚数据确定模型优劣。

## 这版可以做什么

| 输入/实验 | 已实现 |
|---|---|
| 光谱 | ROI中位数/IQR/SNV → 1D CNN |
| 图像 | RGB、绿色通道或CLAHE → ResNet18；可选ImageNet初始化 |
| 联合输入 | 光谱CNN与ResNet18双分支 → 拼接 → MLP分类头 |
| 传统对照 | Logistic Regression、PLS-DA、RBF-SVM、Extra Trees |
| 传统模型评估 | 开发集嵌套批次CV，折内调参与分组校准，可选批次内标签置乱 |
| 深度模型评估 | 外层批次CV，内层独立早停/校准；支持多随机种子 |
| 统计分析 | AUC、平衡准确率、Macro-F1、MCC、Brier、逐性别召回率/精确率、批次bootstrap、配对模型比较 |

默认科研分类采用0.5阈值，对有效输入给出F/M与雌性概率。质量不合格的输入单独记录，不与低概率或错误分类混淆。原有SVM可通过 `decision_policy: "selective"` 启用复检阈值；它是可选分析，不是开始科研的前提。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# CPU版深度模型；GPU服务器请按PyTorch官方方式配置匹配CUDA的版本
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
```

Windows PowerShell 使用虚拟环境 Scripts 目录中的 Activate.ps1。也可以在配置好PyTorch后使用 `python -m pip install -e ".[deep]"` 检查深度模型依赖。

## 训练新的模型

按[数据契约](docs/DATA_CONTRACT.md)准备清单和设备配置。仍使用 `train/val/calib/test` 四个批次隔离的分区。

```bash
# 光谱CNN
python -m eggsex.cli deep-train --manifest data/manifest.csv --config configs/instrument.json --mode hsi --out runs/cnn_hsi
# ImageNet初始化的ResNet18
python -m eggsex.cli deep-train --manifest data/manifest.csv --config configs/instrument.json --mode rgb --variant rgb --pretrained --out runs/resnet_rgb
# 双分支融合模型
python -m eggsex.cli deep-train --manifest data/manifest.csv --config configs/instrument.json --mode fusion --pretrained --out runs/fusion
```

GPU服务器加入 `--device cuda`。不提供 `--pretrained` 时从随机权重开始，不会自动下载权重；使用时需要下载官方ImageNet权重，失败不会静默退回随机初始化。小样本可比较 `--pretrained --freeze-image` 与微调方案。

`deep-train` 只打开开发集文件，按验证损失选最佳epoch，再用calib拟合温度参数。**不会自动测试test。** 输出 `model.pt`、`history.csv`、`metrics.json` 和QC清单。

## 系统比较

```bash
# 四种传统模型 × 三种模态 × 三个种子；最终test文件不打开
python -m eggsex.cli benchmark --manifest data/manifest.csv --config configs/instrument.json --out runs/benchmark
# 三个深度模型的外层批次交叉验证
python -m eggsex.cli deep-cv --manifest data/manifest.csv --config configs/instrument.json --modes hsi,rgb,fusion --seeds 42,123,2026 --pretrained --device cuda --out runs/deep_cv
```

默认外层5折。深度CV需要每个外层训练部分至少有3个批次，以划分拟合、早停和校准。每个相应分区必须含两类。批次不足时可先用固定划分探索，不要把同批样本伪造为多个批次。

`benchmark --permutations 999` 可进行批次内标签置乱，每次重新运行嵌套调参/校准；计算量较大。默认不运行置乱，不能将未运行写成显著性证据。深度CV当前未自动实现完整重训练的置乱检验。

每种模态使用同一套有效鸡蛋。更换预处理、模型家族或超参数并观察OOF结果，仍属于开发探索；确定方案后使用独立test评估。传统与深度模型的内层训练预算不同，比较时需报告这一差异。

## 固定模型后的评估、推理和配对比较

`test.csv` 只包含最终测试鸡蛋，路径相对于它所在目录。评估需要sex真值，推理可以没有sex。

```bash
python -m eggsex.cli deep-evaluate --checkpoint runs/fusion/model.pt --manifest data/test.csv --out runs/final_test
python -m eggsex.cli deep-predict --checkpoint runs/fusion/model.pt --manifest data/unlabeled.csv --out runs/inference
python -m eggsex.cli compare --a runs/fusion_test/predictions.csv --b runs/rgb_test/predictions.csv --out runs/paired_comparison.json
```

评估会检查与开发集的egg_id和batch_id重叠。`compare` 按egg_id对齐、检查标签与批次一致性，报告共同样本数及A−B的准确率差异批次bootstrap区间；比较多种方案仍需处理多重选择问题。

## 说明与验证

- [全面科研评估与模型选择](docs/科研评估_v0.2.md)：论文证据、旧版问题、消融矩阵、统计分析和实验建议。
- [模型卡](docs/MODEL_CARD.md)：算法与现有局限。
- [测试记录](docs/verification_v0.2.json)：实际执行过的工程检查。
- `python -m eggsex.cli demo --out runs/synthetic` 保留旧版SVM演示。

当前仍是一蛋一个预选观察时点/视角。模型直接学习图像，但HSI分支使用ROI汇总谱，尚未学习完整立方体的像素级空间关系。未实现自动分割、Grad-CAM、跨相机配准、3D-CNN/Transformer复现或真实鸡胚权重。原始数据、标签、权重和运行结果留在研究数据存储，不提交Git；代码MIT授权。
