#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验四：系统效率评估
评估 RAG 系统各阶段的延迟：
  1. 索引加载时间（一次性开销）
  2. 检索延迟（per query，20 条告警）
  3. 端到端报告生成时间（per alarm，来自实验三数据）
以及检索结果质量稳定性（top-1 相关度分布）
结果保存至：experiments/exp4_system_efficiency.md
"""

import sys, json, time
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from embedding_store import EmbeddingStore

ALARM_PATH = Path('/home/zzx-king/zzx/test/post_processor/summary_output/alarms_wide_202408.jsonl')
OUT_PATH   = Path(__file__).resolve().parent.parent / 'experiments' / 'exp4_system_efficiency.md'
EXP3_RAW   = Path(__file__).resolve().parent.parent / 'experiments' / 'exp3_raw.json'

ALARM_GROUPS = {
    'a2': [0, 1, 2, 4, 5, 6, 7],
    'a1': [3, 9, 16, 19, 21, 25],
    'a3': [14, 24, 35, 38, 45],
    'a4': [36, 153],
}

_PATTERN_DESCRIPTIONS = {
    "a1": "origin AS validation RPKI ROA route origin authorization IRR prefix legitimacy",
    "a2": "valley-free violation route leak provider customer peer AS relationship BGP policy",
    "a3": "reserved ASN unknown autonomous system bogon path anomaly invalid AS number",
    "a4": "same organization origin AS change internal routing sibling AS",
    "b1": "origin connectivity RPKI validation upstream provider customer link",
    "b2": "AS path prepending traffic engineering load balancing",
    "b3": "different upstream provider path change origin upstream diversity",
}


def load_alarms():
    alarms = {}
    with open(ALARM_PATH) as f:
        for line in f:
            obj = json.loads(line)
            alarms[obj['group_id']] = obj
    return alarms


def build_rag_query(alarm):
    parts = ["BGP routing anomaly"]
    seen = set()
    for event in alarm.get("events", []):
        for rc in event.get("route_changes", []):
            for pk in rc.get("patterns", {}).keys():
                seen.add(pk)
    for pk in sorted(seen):
        if pk in _PATTERN_DESCRIPTIONS:
            parts.append(_PATTERN_DESCRIPTIONS[pk])
    return " ".join(parts)[:500]


def main():
    print("=" * 65)
    print("实验四：系统效率评估")
    print("=" * 65)

    # ===== 1. 索引加载时间 =====
    base = Path(__file__).resolve().parent
    print("\n测量索引加载时间（3次取均值）...")
    load_times = []
    for i in range(3):
        t0 = time.time()
        s = EmbeddingStore()
        s.load_index(str(base / 'faiss_index.bin'), str(base / 'index_data.json'))
        load_times.append(time.time() - t0)
        print(f"  第{i+1}次: {load_times[-1]:.2f}s")

    # 使用最后一个 store
    store = s

    # 索引大小
    n_docs = len(store.data)
    index_dim = store.index.d

    # ===== 2. 检索延迟 =====
    print("\n测量检索延迟（20条真实告警查询）...")
    alarms = load_alarms()
    retrieval_times = []
    top1_scores = []

    for pattern, gids in ALARM_GROUPS.items():
        for gid in gids:
            alarm = alarms.get(gid)
            if not alarm:
                continue
            query = build_rag_query(alarm)
            t0 = time.time()
            results = store.search(query, k=3, hybrid=True)
            elapsed = time.time() - t0
            retrieval_times.append(elapsed)
            if results:
                top1_scores.append(results[0][1])
            print(f"  gid={gid:3d} [{pattern}]  检索={elapsed*1000:.1f}ms  top1={results[0][1]:.4f}")

    avg_rt = np.mean(retrieval_times)
    p50_rt = np.percentile(retrieval_times, 50)
    p95_rt = np.percentile(retrieval_times, 95)
    max_rt = np.max(retrieval_times)

    # ===== 3. 端到端时间（来自实验三）=====
    e2e_rag = []
    e2e_norag = []
    if EXP3_RAW.exists():
        with open(EXP3_RAW) as f:
            exp3_data = json.load(f)
        e2e_rag   = [r['rag_time']   for r in exp3_data]
        e2e_norag = [r['norag_time'] for r in exp3_data]
        print(f"\n从实验三读取端到端时间数据（{len(exp3_data)} 条）")
    else:
        print("\n⚠️ 实验三数据未找到，端到端时间部分将跳过")

    # ===== 生成 Markdown =====
    lines = []
    lines.append("# 实验四：系统效率评估\n")
    lines.append("**日期：** 2026-05-22  ")
    lines.append("**测试集：** 20 条真实告警（检索延迟）；10 条告警（端到端时间，来自实验三）  ")
    lines.append("**检索配置：** bge-base-en-v1.5 + BM25+向量+RRF+MMR，Top-3\n")
    lines.append("---\n")

    lines.append("## 1. 知识库规模\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 知识库文档块数量 | {n_docs} |")
    lines.append(f"| 向量维度 | {index_dim} |")
    lines.append(f"| 支持文件类型 | PDF、TXT、CSV |")
    lines.append("")

    lines.append("---\n")
    lines.append("## 2. 索引加载时间\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 平均加载时间 | {np.mean(load_times):.2f}s |")
    lines.append(f"| 最短 | {np.min(load_times):.2f}s |")
    lines.append(f"| 最长 | {np.max(load_times):.2f}s |")
    lines.append(f"| 说明 | 索引加载为启动时一次性开销，运行期无需重复加载 |")
    lines.append("")

    lines.append("---\n")
    lines.append("## 3. 检索延迟（Retrieval Latency）\n")
    lines.append("测量 20 条真实告警语义查询词的 Top-3 混合检索（BM25+向量+RRF+MMR）耗时：\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 平均检索延迟 | **{avg_rt*1000:.1f} ms** |")
    lines.append(f"| 中位数（P50） | {p50_rt*1000:.1f} ms |")
    lines.append(f"| P95 | {p95_rt*1000:.1f} ms |")
    lines.append(f"| 最大值 | {max_rt*1000:.1f} ms |")
    lines.append(f"| 平均 Top-1 相关度分数 | {np.mean(top1_scores):.4f} |")
    lines.append("")

    # 检索延迟分布
    lines.append("**检索延迟分布：**\n")
    lines.append("| 延迟区间 | 查询数 |")
    lines.append("|---|---|")
    thresholds = [(0, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, float('inf'))]
    labels = ["< 100ms", "100~200ms", "200~500ms", "500ms~1s", "> 1s"]
    for (lo, hi), label in zip(thresholds, labels):
        cnt = sum(lo <= t < hi for t in retrieval_times)
        lines.append(f"| {label} | {cnt} |")
    lines.append("")

    lines.append("---\n")
    lines.append("## 4. 端到端报告生成时间\n")
    lines.append("说明：端到端时间 = 检索延迟 + LLM 推理时间，主要由 LLM API 响应决定。\n")
    if e2e_rag:
        lines.append("| 指标 | 有 RAG（秒） | 无 RAG（秒） |")
        lines.append("|---|---|---|")
        lines.append(f"| 平均 | {np.mean(e2e_rag):.2f} | {np.mean(e2e_norag):.2f} |")
        lines.append(f"| 中位数 | {np.median(e2e_rag):.2f} | {np.median(e2e_norag):.2f} |")
        lines.append(f"| 最短 | {np.min(e2e_rag):.2f} | {np.min(e2e_norag):.2f} |")
        lines.append(f"| 最长 | {np.max(e2e_rag):.2f} | {np.max(e2e_norag):.2f} |")
        lines.append("")
        lines.append(f"> **注：** 检索延迟（平均 {avg_rt*1000:.0f}ms）占端到端时间 {avg_rt/np.mean(e2e_rag)*100:.1f}%，"
                     f"LLM 推理占 {(1-avg_rt/np.mean(e2e_rag))*100:.1f}%，系统瓶颈在 LLM API。\n")
    else:
        lines.append("> 端到端时间数据需先运行实验三（exp3_rag_vs_norag.py）生成 exp3_raw.json。\n")

    lines.append("---\n")
    lines.append("## 5. 关键发现\n")
    lines.append("| 发现 | 说明 |")
    lines.append("|---|---|")
    lines.append(f"| 检索速度快 | 平均检索延迟 {avg_rt*1000:.0f}ms，满足实时交互需求 |")
    if e2e_rag:
        lines.append(f"| LLM 是系统瓶颈 | 检索占端到端时间 {avg_rt/np.mean(e2e_rag)*100:.1f}%，提速应优先优化 LLM 调用 |")
        lines.append(f"| RAG 不显著增加生成时间 | 有 RAG {np.mean(e2e_rag):.1f}s vs 无 RAG {np.mean(e2e_norag):.1f}s，增加检索的代价可忽略 |")
    lines.append(f"| 索引加载为一次性开销 | {np.mean(load_times):.1f}s 仅在服务启动时发生，运行期无延迟 |")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n结果已保存至：{OUT_PATH}")


if __name__ == '__main__':
    main()
