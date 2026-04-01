# Dataset Recipe

这份文档定义当前使用的 `dataset_recipe` 结构。

## 作用

- 把数据集入口、切分和任务路径收敛成结构化对象
- 当前分类训练链路使用它保存数据集视图

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

## 当前分类口径

- 分类训练使用 manifest 文件
- manifest 每行格式为 `relative/path/to/image<TAB>class_name`
- `ExperimentConfig` 在缺省 `dataset_recipe` 时会回填：
  - `source.root_dir = data/raw/<dataset>`
  - `splits.train_manifest = data/classification/<dataset>/train.txt`
  - `splits.val_manifest = data/classification/<dataset>/val.txt`
  - `splits.test_manifest = data/classification/<dataset>/test.txt`
- 如果 `class_names` 为空，系统会从已有 manifest 推断

## 当前数据目录

```text
data/
  raw/
    <dataset_name>/
  classification/
    <dataset_name>/
      train.txt
      val.txt
      test.txt
```

## 当前校验口径

- `task_type == classification`
- `dataset_name` 非空
- `class_names` 非空或可从 manifest 推断
- `train_manifest` 和 `val_manifest` 可用
