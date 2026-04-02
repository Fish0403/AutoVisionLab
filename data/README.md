# Data Guide

这个目录存放数据准备脚本、原始数据和分类训练使用的 manifest。

## 目录结构

```text
data/
  README.md
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

## NEU-CLS 切分

`neu` 的最新原始结构是单层文件夹，文件名里包含类别前缀，例如：

```text
data/raw/NEU-CLS/
  Cr_161.bmp
  In_001.bmp
  ...
```

因此可以直接运行专用切分脚本：

```bash
python3 data/prepare_neucls_split.py \
  --source-root data/raw/NEU-CLS \
  --dataset-name NEU \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42 \
  --force
```

脚本：

- [prepare_neucls_split.py](prepare_neucls_split.py)

这一步会直接生成：

```text
data/classification/NEU/
  train.txt
  val.txt
  test.txt
```

## 当前仓库中的数据
