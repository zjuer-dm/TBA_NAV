#!/bin/bash

# 开启严格错误处理模式
set -e

# 1. 定义并切换至目标数据目录
WORK_DIR="data/datasets"
if [ ! -d "$WORK_DIR" ]; then
    echo "错误：未找到目标目录 $WORK_DIR，请检查当前是否在项目根目录下执行。"
    exit 1
fi
cd "$WORK_DIR"

# 2. 清理旧的解压文件（根据要求移除之前生成的错误目录）
echo "正在清理历史目录..."
rm -rf rxr envdrop r2r

# 3. 处理 RxR 数据集
if [ -f "RxR_VLNCE_v0.zip" ]; then
    echo "解压 RxR_VLNCE_v0.zip..."
    unzip -q RxR_VLNCE_v0.zip
    mv RxR_VLNCE_v0 rxr
else
    echo "错误：未找到 RxR_VLNCE_v0.zip"
fi

# 4. 处理 R2R 数据集 (严格按照 Data Preparation 章节)
if [ -f "R2R_VLNCE_v1.zip" ]; then
    echo "解压 R2R_VLNCE_v1.zip..."
    unzip -q R2R_VLNCE_v1.zip
    mv R2R_VLNCE_v1 r2r
else
    echo "错误：未找到 R2R_VLNCE_v1.zip"
fi

# 5. 处理 EnvDrop 数据集
if [ -f "R2R_VLNCE_v1-3_preprocessed.zip" ]; then
    echo "解压 R2R_VLNCE_v1-3_preprocessed.zip (提取 envdrop)..."
    unzip -q R2R_VLNCE_v1-3_preprocessed.zip
    mv R2R_VLNCE_v1-3_preprocessed/envdrop ./envdrop
    # 删除多余的预处理文件夹主体
    rm -rf R2R_VLNCE_v1-3_preprocessed
else
    echo "错误：未找到 R2R_VLNCE_v1-3_preprocessed.zip"
fi

# 6. 验证最终目录拓扑
echo "数据处理完成，当前 $WORK_DIR 目录结构如下："
ls -ld r2r rxr envdrop scalevln 2>/dev/null || true