#!/bin/bash

# 检查参数
if [ "$#" -ne 3 ]; then
    echo "用法: $0 <collector> <year> <month>"
    echo "例如: $0 wide 2024 8"
    exit 1
fi

collector="$1"
year="$2"
month="$3"

echo "生成报告..."

# 运行summary_routeviews.py脚本
echo "运行summary_routeviews.py脚本..."
python3 post_processor/summary_routeviews.py -c "$collector" -y "$year" -m "$month"

echo "报告生成完成！"
echo "报告路径: post_processor/html/report_${collector}_${year}${month:02}.html"
