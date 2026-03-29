# Dataset Recipe Schema

这份文档定义 AutoVisionLab 后续引入的 `dataset_recipe` 结构。

## 1. 目标

`dataset_recipe` 的作用是把数据集入口、切分和任务相关路径收敛成结构化对象。

`v1` 范围明确限制为：

- 任务类型：`classification`
- 数据入口继续兼容当前 manifest 方案

同时，这份 schema 从第一天起就为后续检测和分割预留顶层结构。

## 2. 设计原则

- `v1` 不推翻现有分类 manifest 方案
- 顶层统一保留 `task_type`
- 数据 schema 应能支持 artifact 回写和 API 返回
- 后续检测 / 分割不再另起一套完全独立的数据描述对象

## 3. 顶层结构

建议的 `dataset_recipe.yaml` 如下：

```yaml
version: dataset_recipe@v1
task_type: classification
dataset_name: neu
class_names:
  - crazing
  - inclusion
  - patches
  - pitted_surface
  - rolled_in_scale
  - scratches
source:
  root_dir: data/raw/neu
  prepared_source_dir: data/classification/neu/classification_source
splits:
  train_manifest: data/classification/neu/train.txt
  val_manifest: data/classification/neu/val.txt
  test_manifest: data/classification/neu/test.txt
metadata:
  image_size_options: [32, 64, 96]
  notes: null
```

## 4. `v1` 分类字段

### 4.1 基本字段

- `version`
  - 固定为 `dataset_recipe@v1`
- `task_type`
  - `v1` 固定为 `classification`
- `dataset_name`
- `class_names`

### 4.2 `source`

建议字段：

- `root_dir`
- `prepared_source_dir`

说明：

- `root_dir` 指原始或主数据入口
- `prepared_source_dir` 指 manifest 实际引用的图像根目录

### 4.3 `splits`

`v1` 分类使用 manifest：

- `train_manifest`
- `val_manifest`
- `test_manifest`

原因：

- 当前训练器已经围绕 manifest 读取实现
- 这比一开始改成目录扫描更符合当前平台状态

## 5. 与当前实现的关系

当前分类训练器的数据入口在这里：

- [backend/app/trainers/classification/base_trainer.py](../../backend/app/trainers/classification/base_trainer.py)

当前数据目录规则见：

- [../policies/experiment_policy.md](../policies/experiment_policy.md)
- [data/README.md](../data/README.md)

后续建议演进为：

- 训练器不再直接拼路径
- 改成先读取 `dataset_recipe`
- 再由 `task_type` 选择对应的数据解释器

## 6. 面向未来多任务的预留结构

虽然 `v1` 只做分类，但建议在设计上预留下面两种形态。

### 6.1 detection 示例

```yaml
version: dataset_recipe@v1
task_type: detection
dataset_name: sample_det
class_names: [defect]
splits:
  train_images: data/detection/sample_det/images/train
  train_labels: data/detection/sample_det/labels/train
  val_images: data/detection/sample_det/images/val
  val_labels: data/detection/sample_det/labels/val
```

### 6.2 segmentation 示例

```yaml
version: dataset_recipe@v1
task_type: segmentation
dataset_name: sample_seg
class_names: [background, defect]
splits:
  train_images: data/segmentation/sample_seg/images/train
  train_masks: data/segmentation/sample_seg/masks/train
  val_images: data/segmentation/sample_seg/images/val
  val_masks: data/segmentation/sample_seg/masks/val
```

说明：

- 这两种结构只是未来预留，不代表当前已经支持
- 当前实现仍然只有分类链路

## 7. `v1` 推荐校验规则

分类场景下建议至少校验：

- `task_type == classification`
- `dataset_name` 非空
- `class_names` 非空
- `train_manifest` 和 `val_manifest` 存在
- manifest 中的类别必须都属于 `class_names`

## 8. API / artifact 建议

后续建议：

- `ExperimentConfig` 增加 `dataset_recipe`
- 每次实验把 `dataset_recipe` 落盘到：
  - `artifacts/recipes/<experiment_id>_dataset.yaml`
- 结果页提供数据 recipe 快照查看入口

## 9. 一句话总结

`dataset_recipe@v1` 的目标不是替换当前分类 manifest，而是先把它包进统一的数据对象里，为后续检测和分割接入同一平台打基础。
