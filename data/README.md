# Data Guide

这个目录存放数据准备脚本、原始数据和分类训练使用的 manifest。

## 目录结构

```text
data/
  README.md
  prepare_classification_split.py
  raw/
    <dataset_name>/
  classification/
    <dataset_name>/
      train.txt
      val.txt
      test.txt
```

约定：

- `raw/` 存按类别名分文件夹的原始图像或轻量预处理结果
- `classification/<dataset_name>/` 存分类任务使用的 split manifest
- manifest 每行格式为 `relative/path/to/image<TAB>class_name`
- 路径相对于对应数据集的 source 根目录

## 通用分类切分脚本

脚本：

- [prepare_classification_split.py](prepare_classification_split.py)

适用前提：

- 输入源必须是按类别分文件夹的结构

常用命令：

```bash
python3 data/prepare_classification_split.py \
  --source-dir data/raw/your_dataset \
  --dataset-name your_dataset \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42
```

如果输入目录名本身就想作为输出名，可以省略 `--dataset-name`。

如果数据本来就在 `data/raw/<dataset_name>/` 下，也可以只传名字。

默认行为：

- 输出固定落到 `data/classification/<name>/`
- 只生成 `train.txt`、`val.txt`、`test.txt`
- 不复制原始图片
- 按类别分层随机切分
- 保留类别目录下的嵌套子目录结构
- 如果输出已存在，需要加 `--force`

## NEU-CLS 预处理

`neu` 的原始结构为：

```text
data/raw/neu/
  NEU-CLS/
    train/train/images/
    valid/valid/images/
```

因此在运行通用 split 之前，需要先执行预处理：

```bash
python3 data/raw/neu/prepare_classification_source.py --force
```

预处理脚本：

- [prepare_classification_source.py](raw/neu/prepare_classification_source.py)

这一步会生成：

```text
data/raw/neu/classification_source/
  crazing/
  inclusion/
  patches/
  pitted_surface/
  rolled-in_scale/
  scratches/
```

之后再运行通用 split 脚本。

## 当前仓库中的数据
