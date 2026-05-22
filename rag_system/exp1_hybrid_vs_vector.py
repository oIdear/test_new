#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验一：混合检索 vs 纯向量检索对比
对 20 条标准 BGP 领域查询，比较：
  - 混合检索（BM25 + 向量 + RRF + MMR，hybrid=True）
  - 纯向量检索（FAISS 余弦相似度，hybrid=False）
评估指标：Top-3 召回率、来源文件多样性
结果保存至：experiments/exp1_hybrid_vs_vector.md（数据已内嵌）
"""

import sys, json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from embedding_store import EmbeddingStore

OUT_PATH = Path(__file__).parent.parent / 'experiments' / 'exp1_raw.json'

# 20 条覆盖各 pattern 类型的标准 BGP 查询
QUERIES = [
    # a1: RPKI / ROA / IRR
    ("RPKI ROA route origin authorization prefix legitimacy",               ["RPKI", "ROA", "origin"]),
    ("origin AS validation IRR internet routing registry",                  ["IRR", "routing", "origin"]),
    ("BGP route origin validation prefix legitimacy RPKI ROV",              ["RPKI", "origin", "valid"]),
    # a2: valley-free / route leak
    ("valley-free violation route leak provider customer peer",             ["valley", "provider", "customer"]),
    ("route leak BGP policy propagation AS relationship",                   ["route leak", "leak", "provider"]),
    ("BGP route leak detection prevention RFC 7908",                        ["route leak", "leak", "provider"]),
    # a3: reserved / unknown ASN
    ("reserved ASN bogon invalid AS number in path",                        ["reserved", "bogon", "AS"]),
    ("unknown autonomous system number path anomaly",                       ["unknown", "AS", "path"]),
    # a4: same organization
    ("same organization origin AS change sibling AS",                       ["organization", "origin", "AS"]),
    # b1: origin connectivity
    ("origin connectivity upstream provider customer link RPKI",            ["provider", "customer", "origin"]),
    # b2: AS prepend
    ("AS path prepending traffic engineering load balancing",               ["prepend", "AS_PATH", "traffic"]),
    # b3: different upstream
    ("different upstream provider path change origin diversity",            ["upstream", "provider", "path"]),
    # 通用 BGP 安全
    ("BGP prefix hijack detection mitigation",                              ["hijack", "origin", "prefix"]),
    ("BGP security routing anomaly detection",                              ["BGP", "routing", "anomaly"]),
    ("AS path attribute BGP UPDATE message",                                ["AS_PATH", "UPDATE", "path"]),
    ("BGP route selection decision process",                                ["Decision", "preference", "route"]),
    ("RPKI ROA validation not-found invalid valid status",                  ["RPKI", "valid", "invalid"]),
    ("BGP route leak RFC 9234 BGP roles OTC attribute",                     ["route leak", "roles", "OTC"]),
    ("RPKI resource public key infrastructure origin validation RFC 6811",  ["RPKI", "origin", "validation"]),
    ("extreme path anomaly unseen AS relationship",                         ["anomaly", "path", "AS"]),
]


def recall_hit(results, keywords):
    return any(
        any(kw.lower() in item['content'].lower() for kw in keywords)
        for item, _ in results
    )


def source_diversity(results):
    return len(set(item.get('file', '?') for item, _ in results))


def main():
    print("=" * 65)
    print("实验一：混合检索 vs 纯向量检索")
    print("=" * 65)

    store = EmbeddingStore()
    store.load_index(
        str(Path(__file__).parent / 'faiss_index.bin'),
        str(Path(__file__).parent / 'index_data.json'),
    )

    rows = []
    for query, kws in QUERIES:
        r_hyb  = store.search(query, k=3, hybrid=True)
        r_pure = store.search(query, k=3, hybrid=False)

        rows.append({
            'query':    query[:50],
            'kws':      kws,
            'hit_h':    recall_hit(r_hyb,  kws),
            'div_h':    source_diversity(r_hyb),
            'score_h':  round(r_hyb[0][1],  4) if r_hyb  else 0.0,
            'files_h':  [r[0].get('file', '?') for r in r_hyb],
            'hit_p':    recall_hit(r_pure, kws),
            'div_p':    source_diversity(r_pure),
            'score_p':  round(r_pure[0][1], 4) if r_pure else 0.0,
            'files_p':  [r[0].get('file', '?') for r in r_pure],
        })

    # ===== 汇总 =====
    n = len(rows)
    hits_h  = sum(r['hit_h']  for r in rows)
    hits_p  = sum(r['hit_p']  for r in rows)
    avg_dh  = np.mean([r['div_h']  for r in rows])
    avg_dp  = np.mean([r['div_p']  for r in rows])

    print(f"\n{'查询（前30字符）':<32} {'混合命中':<8} {'混合多样性':<10} {'向量命中':<8} {'向量多样性'}")
    print("-" * 70)
    for r in rows:
        mh = '✓' if r['hit_h'] else '✗'
        mp = '✓' if r['hit_p'] else '✗'
        print(f"{r['query'][:30]:<32} {mh:<8} {r['div_h']:<10} {mp:<8} {r['div_p']}")

    print("\n" + "=" * 65)
    print(f"{'指标':<30} {'混合检索':>15} {'纯向量检索':>15}")
    print("-" * 62)
    print(f"{'Top-3 命中数（/20）':<30} {hits_h:>15} {hits_p:>15}")
    print(f"{'Top-3 召回率':<30} {hits_h/n:>14.1%} {hits_p/n:>14.1%}")
    print(f"{'平均来源文件数（多样性）':<30} {avg_dh:>15.2f} {avg_dp:>15.2f}")
    print("=" * 65)

    # ===== 多样性分布 =====
    from collections import Counter
    cnt_h = Counter(r['div_h'] for r in rows)
    cnt_p = Counter(r['div_p'] for r in rows)
    print("\n来源多样性分布（每查询3个结果来自几个不同文件）：")
    print(f"{'多样性值':<10} {'混合检索（条）':>14} {'纯向量检索（条）':>16}")
    for v in [1, 2, 3]:
        print(f"{v:<10} {cnt_h.get(v, 0):>14} {cnt_p.get(v, 0):>16}")

    # 保存原始数据
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n原始数据已保存至：{OUT_PATH}")


if __name__ == '__main__':
    main()
