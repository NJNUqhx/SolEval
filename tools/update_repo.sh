#!/bin/bash
set -e  # 脚本出错自动停止

echo "===== 开始同步 repository 文件夹 ====="

# 1. 判断当前目录是否存在 repository，存在则删除
if [ -d "repository" ]; then
    echo "🗑️  删除旧的 repository 文件夹..."
    rm -r repository
else
    echo "ℹ️  当前目录无 repository，直接复制"
fi

# 2. 判断上级目录是否存在 repository，存在则复制
if [ -d "../repository" ]; then
    echo "📂 复制上级目录的 repository 到当前..."
    cp -r ../repository .
    echo "✅ 同步完成！"
else
    echo "❌ 错误：上级目录未找到 repository 文件夹"
    exit 1
fi