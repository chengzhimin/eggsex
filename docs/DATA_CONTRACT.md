# 数据契约

## Manifest

```csv
egg_id,batch_id,sex,split,device_id,protocol_id,age_h,hsi_path,rgb_path,rgb_mask_path
B01_E001,B01,F,train,HSI01,P01,90,eggs/B01_E001.npz,rgb/B01_E001.png,masks/B01_E001.png
```

`egg_id` 是样本级唯一 ID；`batch_id` 用于阻止批次泄漏；`sex` 仅表示外部获得的真值标签。训练集必须使用 `train/val/calib/test` 四个分区，且同一批次只能进入一个分区。正式评价前禁止依据 test 结果选择模型或阈值。

## Spectral NPZ

- `raw`: H×W×B 原始 DN
- `dark`: 可广播的暗参考
- `white`: 可广播的透射参考
- `wavelengths_nm`: B 个实际中心波长，递增、单位 nm
- `mask`: H×W 布尔 ROI
- `saturation_dn`: 实际饱和 DN

参考数据必须与采集条件匹配。波长不满足配置容差时，程序拒绝输入而不自动插值。大型立方体后续应改为 memmap/分块读取。

## Image

当前图像分支要求 8-bit 三通道图像和对齐 ROI mask；优先保存无损原始图，已有JPEG需报告压缩来源。12/16-bit 数据应增加明确的量化适配器，而不是静默压缩。

## 数据隔离

不要把原始数据、真值标签、模型权重和 `runs/` 输出提交到 Git。跨时间点或多视角数据必须在 split 前按 `egg_id` 聚合，不能通过复制行扩大样本量。

## v0.2补充

科研默认输出0.5阈值分类及概率；可选decision_policy=selective仅用于旧SVM的选择性分析。deep-train仅读取train/val/calib，deep-evaluate需要单独的test.csv；不必为了运行训练提前查看test影像。

benchmark/deep-cv在开发批次上生成OOF结果，排除最终test文件。同一个运行中的所有模态用共同QC人群，排除清单保留；跨运行比较仍应核对cohort.csv中的egg_id是否一致。

配置同时提供age_min_h和age_max_h时，数据必须有有限数值age_h并位于区间内。未提供边界时不限制胚龄；论文仍需报告实际年龄分布。当前一个配置对应一个设备/协议，跨设备实验需适配。

路径仅用于读取，不作为模型特征；不同egg_id不得引用同一次真实观测来伪造独立样本。当前检查ID与批次，不保证检测修改ID后的内容级重复。mask仍由外部提供，不含自动标注。

RGB分支实际可以读取JPEG，但应优先保存无损原始图；已有JPEG不因格式本身自动判废，需报告压缩来源和影响。
