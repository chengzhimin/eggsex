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

当前图像分支要求 8-bit 三通道无损图像和对齐 ROI mask。JPEG 不作为训练原始数据。12/16-bit 数据应增加明确的量化适配器，而不是静默压缩。

## 数据隔离

不要把原始数据、真值标签、模型权重和 `runs/` 输出提交到 Git。跨时间点或多视角数据必须在 split 前按 `egg_id` 聚合，不能通过复制行扩大样本量。
