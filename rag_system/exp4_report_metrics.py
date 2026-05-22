#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6.3.3 结构化报告指标测量
对 20 条告警调用完整的 RAG 分析流程，统计：
1. 报告生成时间（从发起请求到收到完整响应）
2. 每份报告中的可溯源引用标注数（[来源:] 出现次数）
"""

import sys
import json
import time
import re
import requests
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from embedding_store import EmbeddingStore

DEEPSEEK_API_KEY = "sk-18be0971348e40b6b5ff9ffcfe19a63e"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
N_ALARMS = 10


def load_alarms(n: int) -> list:
    """取前 n 条告警（group_id 0 ~ n-1）。"""
    path = BASE_DIR.parent / "post_processor" / "summary_output" / "alarms_wide_202408.jsonl"
    alarms = []
    with open(path) as f:
        for line in f:
            if len(alarms) >= n:
                break
            alarms.append(json.loads(line))
    return alarms


def build_query(alarm: dict) -> str:
    parts = ["BGP routing anomaly detection"]
    for ev in alarm.get("events", []):
        prefixes = ev.get("prefix", [])
        if prefixes:
            parts.append(f"affected prefix: {prefixes[0]}")
        for rc in ev.get("route_changes", []):
            if rc.get("path1"):
                parts.append(f"path before: {rc['path1']}")
            if rc.get("path2"):
                parts.append(f"path after: {rc['path2']}")
            diff = rc.get("diff")
            if diff is not None and diff != float("inf"):
                parts.append(f"beam_diff_score: {diff:.4f}")
            elif diff == float("inf"):
                parts.append("beam_diff_score: infinity (extreme anomaly)")
            culprit_asns = [
                asn for grp in rc.get("culprit", [])
                if isinstance(grp, list) for asn in grp
            ]
            if culprit_asns:
                parts.append(f"suspect AS: {', '.join(culprit_asns)}")
            patterns = rc.get("patterns", {})
            if "a2" in patterns:
                parts.append("valley_free_violation detected")
            if "a1" in patterns:
                a1 = patterns["a1"]
                for key in ["origin_rpki_1", "origin_rpki_2", "origin_irr_1", "origin_irr_2"]:
                    if a1.get(key):
                        parts.append(f"{key}: {a1[key]}")
            if "a3" in patterns:
                a3 = patterns["a3"]
                if "reserved_path_1" in a3 or "reserved_path_2" in a3:
                    parts.append("reserved_asn_in_path")
                if "unknown_asn_1" in a3 or "unknown_asn_2" in a3:
                    parts.append("unknown_asn detected")
                if "none_rel_1" in a3 or "none_rel_2" in a3:
                    parts.append("no_business_relationship_in_path")
            if "b1" in patterns:
                conn = patterns["b1"].get("origin_connection", "")
                if conn:
                    parts.append(f"origin_connection_type: {conn}")
            if "b2" in patterns:
                parts.append("as_path_prepending detected")
            if "b3" in patterns:
                up = patterns["b3"].get("origin_different_upstream", "")
                if up:
                    parts.append(f"different_upstream_as: {up}")
    parts.append("route leak hijack misconfiguration RPKI IRR ROA valley-free BGP security")
    return " ".join(parts)[:2000]


def build_rag_context(alarm: dict, store: EmbeddingStore, k: int = 3) -> str:
    query = build_query(alarm)
    results = store.search(query, k=k)
    blocks = []
    for item, dist in results:
        blocks.append(
            f"[来源:{item.get('file','unknown')} | 相似度:{dist:.3f}]\n"
            f"{item['content'][:800]}"
        )
    return "\n\n".join(blocks)


PROMPT_TEMPLATE = """
你是一名BGP异常检测与溯源分析专家，同时具备RFC标准、AS关系模型和路由策略分析经验。

现在提供两类信息：
一类是【异常观测数据】（来自BGP检测系统的结构化事件）
一类是【检索增强上下文】（来自RFC文档、研究论文和运维知识库的相关资料）

请基于检索证据进行约束推理，禁止脱离证据臆测结论。
每引用一条检索资料作为判据，必须在句末附注来源，格式严格为：[来源:文件名 | 相似度:X.XXX]

====================
【异常观测数据】
{alarm_json}
====================

【检索增强上下文】
{rag_context}

请输出结构化溯源分析报告，包含：
【1】异常类型判定（路由泄露/路由劫持/错误宣告/策略异常）及判定依据
【2】关键证据链（用"证据→推论"形式，引用RAG资料中的判据）
【3】验证建议（RPKI / IRR / AS关系）
【4】置信度评分（0–1）
"""


def call_deepseek(alarm_json: str, rag_context: str):
    """返回 (response_text, elapsed_seconds)。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {
                "role": "system",
                "content": "你是一个BGP异常溯源分析专家，基于提供的上下文信息对BGP异常进行分析。",
            },
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    alarm_json=alarm_json, rag_context=rag_context
                ),
            },
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }
    t0 = time.time()
    resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    elapsed = time.time() - t0
    content = resp.json()["choices"][0]["message"]["content"]
    return content, elapsed


def count_citations(text: str) -> int:
    """统计文本中 [来源:xxx] 格式的引用标注数量。"""
    return len(re.findall(r"\[来源:[^\]]+\]", text))


def main():
    print("=" * 65)
    print("6.3.3 结构化报告生成指标测量")
    print(f"测试规模：{N_ALARMS} 条告警，使用完整 RAG 分析流程")
    print("=" * 65)

    store = EmbeddingStore()
    store.load_index(
        str(BASE_DIR / "faiss_index.bin"),
        str(BASE_DIR / "index_data.json"),
    )

    alarms = load_alarms(N_ALARMS)
    print(f"加载告警：{len(alarms)} 条\n")

    records = []
    print(f"{'#':<4} {'group_id':<10} {'生成时间(s)':<12} {'引用标注数':<10}")
    print("-" * 40)

    for i, alarm in enumerate(alarms):
        gid = alarm.get("group_id", "?")
        alarm_json = json.dumps(alarm, indent=2, ensure_ascii=False)
        rag_context = build_rag_context(alarm, store, k=3)

        try:
            response, elapsed = call_deepseek(alarm_json, rag_context)
            citations = count_citations(response)
        except Exception as e:
            print(f"{i+1:<4} {str(gid):<10} 调用失败: {e}")
            continue

        records.append({"group_id": gid, "elapsed": elapsed, "citations": citations})
        print(f"{i+1:<4} {str(gid):<10} {elapsed:<12.2f} {citations:<10}")

        time.sleep(0.5)  # 避免API限速

    if not records:
        print("无有效记录，退出。")
        return

    elapsed_vals  = [r["elapsed"]   for r in records]
    citation_vals = [r["citations"] for r in records]

    print("\n" + "=" * 65)
    print("统计汇总")
    print("=" * 65)
    print(f"有效报告数：{len(records)}")
    print()
    print(f"【生成时间（秒）】")
    print(f"  平均值：{np.mean(elapsed_vals):.2f}s")
    print(f"  中位数：{np.median(elapsed_vals):.2f}s")
    print(f"  最小值：{np.min(elapsed_vals):.2f}s")
    print(f"  最大值：{np.max(elapsed_vals):.2f}s")
    print(f"  标准差：{np.std(elapsed_vals):.2f}s")
    print()
    print(f"【每份报告引用标注数 [来源:]】")
    print(f"  平均值：{np.mean(citation_vals):.2f} 条")
    print(f"  中位数：{np.median(citation_vals):.2f} 条")
    print(f"  最小值：{int(np.min(citation_vals))} 条")
    print(f"  最大值：{int(np.max(citation_vals))} 条")
    print(f"  含至少1条引用的报告比例：{sum(c>0 for c in citation_vals)/len(citation_vals):.1%}")
    print("=" * 65)

    output_path = BASE_DIR / "report_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n明细已保存至：{output_path}")


if __name__ == "__main__":
    main()
