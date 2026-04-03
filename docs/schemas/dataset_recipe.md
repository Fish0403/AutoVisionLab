# Dataset Recipe

这份文档定义当前使用的 `dataset_recipe` 结构。

## 作用

- 把数据集入口、切分和训练侧数据视图收敛成结构化对象
- 为 `ExperimentConfig` 提供统一的数据集配置入口
- 为训练器、历史持久化和任务回放提供稳定的数据集快照

## 字段

- `version`
  - 固定为 `dataset_recipe@v1`
- `task_type`
  - 当前固定为 `classification`
- `dataset_name`
- `class_names`
- `source`
- `splits`
- `metadata`

## 子结构

- `source.root_dir`
  - 原始或主数据入口
- `source.prepared_source_dir`
  - manifest 实际引用的图像根目录
- `splits.train_manifest`
- `splits.val_manifest`
- `splits.test_manifest`
  - 可选
- `metadata.image_size_options`
- `metadata.notes`

## 默认构造

当前代码提供默认 `dataset_recipe` 构造 helper，默认值为：

- `source.root_dir = data/raw/<dataset>`
- `splits.train_manifest = data/classification/<dataset>/train.txt`
- `splits.val_manifest = data/classification/<dataset>/val.txt`
- `splits.test_manifest = data/classification/<dataset>/test.txt`

这些值表示默认构造结果；如果调用方显式提供字段，则以显式值为准。

## class_names 推断规则

- 如果 `class_names` 已显式提供，直接使用
- 如果 `class_names` 为空，系统会尝试从已有 manifest 推断
- 当前会读取 `train_manifest`、`val_manifest`，以及存在时的 `test_manifest`
- 如果这些 manifest 都不可用，则 `class_names` 保持为空

## 当前校验口径

- `task_type == classification`
- `dataset_name` 必须与 `ExperimentConfig.dataset` 保持一致
- `train_manifest` 和 `val_manifest` 必须可配置
- `class_names` 可以为空，但只有在已有 manifest 时才会被自动补全
