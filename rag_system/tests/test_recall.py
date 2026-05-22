#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6.2.1 检索召回率测试
从 alarms_wide_202408.jsonl 中选取 20 条覆盖不同 pattern 类型的告警，
使用生产代码 rag_search_context() 构建查询词，评估 Top-3 检索命中率。
"""

import json
import sys
import numpy as np
from pathlib import Path
import faiss

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from embedding_store import EmbeddingStore

# ===== 各 pattern 类型对应的 RFC 相关关键词（ground truth）=====
# a1: Origin验证 (RPKI/IRR/WHOIS)  → RFC 8205 / RFC 4271
# a2: Valley-free 违规              → RFC 4271 / RFC 7908
# a3: 路径异常 (reserved/unknown)   → RFC 4271
# a4: 同源组织                       → RFC 4271
# b1: Origin连接类型与RPKI          → RFC 8205 / RFC 4271
# b2: AS Prepend                     → RFC 4271
# b3: 不同上游                        → RFC 7908 / RFC 4271
PATTERN_GROUND_TRUTH = {
    "a1": ["RPKI", "ROA", "IRR", "origin", "validation"],
    "a2": ["valley", "route leak", "provider", "customer"],
    "a3": ["reserved", "AS_PATH", "AS number", "autonomous system"],
    "a4": ["origin", "autonomous system", "organization"],
    "b1": ["provider", "customer", "peer", "origin", "RPKI"],
    "b2": ["prepend", "AS_PATH", "traffic"],
    "b3": ["upstream", "provider", "leak"],
    "b5": ["route", "BGP", "path"],
}


def dominant_pattern(alarm: dict) -> str:
    """返回告警中出现次数最多的 pattern 类型。"""
    counts: dict = {}
    for ev in alarm.get("events", []):
        for rc in ev.get("route_changes", []):
            for k in rc.get("patterns", {}).keys():
                counts[k] = counts.get(k, 0) + 1
    return max(counts, key=counts.get) if counts else "unknown"


def select_20_diverse_alarms(jsonl_path: Path) -> list:
    """
    从告警文件中按 pattern 类型均衡采样，共选 20 条。
    目标分布（按实际 dominant pattern 分布调整）：
    a2×7, a1×6, a3×5, a4×2  = 20
    """
    targets = {"a2": 7, "a1": 6, "a3": 5, "a4": 2}
    buckets: dict = {k: [] for k in targets}

    with open(jsonl_path) as f:
        for line in f:
            alarm = json.loads(line)
            dp = dominant_pattern(alarm)
            if dp in buckets and len(buckets[dp]) < targets[dp]:
                buckets[dp].append(alarm)

    selected = []
    for k, alarms in buckets.items():
        selected.extend(alarms)
    return selected


def rag_search_context(alarm: dict) -> str:
    """与 app.py 中的生产代码保持一致的查询词构建逻辑。"""
    parts = ["BGP routing anomaly detection"]

    if alarm.get("start_time"):
        parts.append(f"time window: {alarm['start_time']} to {alarm.get('end_time', 'unknown')}")

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

            culprit_list = rc.get("culprit", [])
            culprit_asns = [asn for grp in culprit_list if isinstance(grp, list) for asn in grp]
            if culprit_asns:
                parts.append(f"suspect AS: {', '.join(culprit_asns)}")

            patterns = rc.get("patterns", {})
            if "a1" in patterns:
                a1 = patterns["a1"]
                for key in ["origin_rpki_1", "origin_rpki_2", "origin_irr_1", "origin_irr_2"]:
                    if a1.get(key):
                        parts.append(f"{key}: {a1[key]}")
            if "a2" in patterns:
                parts.append("valley_free_violation detected")
            if "a3" in patterns:
                a3 = patterns["a3"]
                if "reserved_path_1" in a3 or "reserved_path_2" in a3:
                    parts.append("reserved_asn_in_path")
                if "unknown_asn_1" in a3 or "unknown_asn_2" in a3:
                    parts.append("unknown_asn detected")
                if "none_rel_1" in a3 or "none_rel_2" in a3:
                    parts.append("no_business_relationship_in_path")
            if "a4" in patterns:
                same_org = patterns["a4"].get("origin_same_org", "")
                if same_org:
                    parts.append(f"same_organization: {same_org}")
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


def is_hit(retrieved_chunks: list, keywords: list) -> bool:
    """Top-3 块中若有任一块包含至少一个关键词则命中。"""
    for chunk in retrieved_chunks:
        text = chunk.lower()
        if any(kw.lower() in text for kw in keywords):
            return True
    return False


def main():
    alarm_path = BASE_DIR.parent / "post_processor" / "summary_output" / "alarms_wide_202408.jsonl"
    index_path = BASE_DIR / "faiss_index.bin"
    data_path  = BASE_DIR / "index_data.json"

    print("=" * 65)
    print("6.2.1 RAG检索召回率测试")
    print("=" * 65)

    # 加载向量索引
    print("加载FAISS索引...")
    store = EmbeddingStore()
    store.load_index(str(index_path), str(data_path))

    # 选取20条多样化告警
    print("采样告警数据...")
    alarms = select_20_diverse_alarms(alarm_path)
    print(f"共选取 {len(alarms)} 条告警\n")

    hits = 0
    results = []

    for alarm in alarms:
        dp = dominant_pattern(alarm)
        gid = alarm.get("group_id", "?")
        keywords = PATTERN_GROUND_TRUTH.get(dp, ["BGP", "route"])

        query = rag_search_context(alarm)
        retrieved = store.search(query, k=3)
        chunks = [item["content"] for item, _ in retrieved]

        hit = is_hit(chunks, keywords)
        if hit:
            hits += 1

        results.append({
            "group_id": gid,
            "dominant_pattern": dp,
            "hit": hit,
            "keywords": keywords,
            "top1_preview": chunks[0][:120] if chunks else "",
        })

    # ===== 汇总输出 =====
    recall = hits / len(alarms)
    print(f"{'group_id':<12} {'pattern':<8} {'命中':<6} {'验证关键词'}")
    print("-" * 65)
    for r in results:
        mark = "✓" if r["hit"] else "✗"
        kws = ", ".join(r["keywords"][:3])
        print(f"{str(r['group_id']):<12} {r['dominant_pattern']:<8} {mark:<6} {kws}")

    print("\n" + "=" * 65)
    print(f"测试告警总数：{len(alarms)}")
    print(f"Top-3 命中数：{hits}")
    print(f"Top-3 召回率：{recall:.1%}")
    print("=" * 65)

    # 按 pattern 类型统计
    from collections import defaultdict
    by_pattern: dict = defaultdict(lambda: {"total": 0, "hits": 0})
    for r in results:
        by_pattern[r["dominant_pattern"]]["total"] += 1
        if r["hit"]:
            by_pattern[r["dominant_pattern"]]["hits"] += 1
    print("\n各 pattern 类型命中率：")
    for pat, stat in sorted(by_pattern.items()):
        rate = stat["hits"] / stat["total"]
        print(f"  {pat}: {stat['hits']}/{stat['total']}  ({rate:.0%})")


if __name__ == "__main__":
    main()
