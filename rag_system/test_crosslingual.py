#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6.2.2 跨语言检索对比实验
方案A（实际系统）：中文查询 → DeepSeek翻译为英文 → all-MiniLM-L6-v2编码 → FAISS检索
方案B（基线）   ：中文查询 → all-MiniLM-L6-v2直接编码（绕过翻译）→ FAISS检索
评估指标：Top-3召回率、Top-1平均余弦相似度
"""

import sys
import time
import json
import requests
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from embedding_store import EmbeddingStore

import faiss

# ===== 配置 =====
DEEPSEEK_API_KEY = "sk-18be0971348e40b6b5ff9ffcfe19a63e"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# ===== 20条中文查询 + ground truth关键词 =====
# 覆盖路由泄露、劫持、RPKI、AS路径、BGP协议机制等典型运维场景
QUERIES = [
    ("什么是路由泄露，它与路由劫持有何区别？",              ["route leak", "hijack"]),
    ("BGP的valley-free原则是什么？",                        ["valley", "provider", "customer"]),
    ("RPKI如何验证BGP路由的合法性？",                       ["RPKI", "ROA", "origin"]),
    ("BGP UPDATE消息包含哪些字段？",                         ["UPDATE", "path attributes", "NLRI"]),
    ("AS路径属性在BGP中的作用是什么？",                      ["AS_PATH", "path"]),
    ("BGP有限状态机包含哪些状态？",                          ["Idle", "Established", "state"]),
    ("什么情况下会触发BGP NOTIFICATION消息？",               ["NOTIFICATION", "error"]),
    ("BGP邻居之间如何通过OPEN消息建立会话？",                ["OPEN", "session", "TCP"]),
    ("NEXT_HOP属性的作用和规则是什么？",                     ["NEXT_HOP"]),
    ("路由泄露的常见原因和传播机制是什么？",                  ["route leak", "provider", "customer"]),
    ("BGP路由选择决策过程如何工作？",                        ["Decision", "preference"]),
    ("如何通过ROA防止BGP路由劫持？",                         ["ROA", "origin", "RPKI"]),
    ("BGP KEEPALIVE消息的作用和定时器配置？",                ["KEEPALIVE", "timer"]),
    ("AS路径的prepend操作如何影响流量？",                    ["prepend", "AS_PATH"]),
    ("IRR数据库在路由安全中起什么作用？",                    ["IRR", "routing", "route"]),
    ("如何识别和响应BGP路由劫持事件？",                      ["hijack", "origin", "path"]),
    ("LOCAL_PREF属性如何影响BGP路由选择？",                  ["LOCAL_PREF", "preference"]),
    ("BGP路由聚合的作用和ATOMIC_AGGREGATE属性？",            ["aggregat", "ATOMIC_AGGREGATE"]),
    ("valley-free违规是如何导致路由泄露的？",                ["valley", "route leak", "provider"]),
    ("BGP MULTI_EXIT_DISC属性的使用场景？",                  ["MULTI_EXIT_DISC", "MED"]),
]


def translate_to_english(chinese_text: str) -> str:
    """调用DeepSeek API将中文查询翻译为英文。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的技术翻译助手。请将用户的中文技术问题准确翻译为英文，保持专业术语的准确性。只返回翻译结果，不要添加任何解释。",
            },
            {"role": "user", "content": f"请将以下中文技术问题翻译为英文：\n\n{chinese_text}"},
        ],
        "temperature": 0.1,
        "max_tokens": 200,
    }
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  ⚠ 翻译失败: {e}，返回原文")
        return chinese_text


def search_with_score(store: EmbeddingStore, query_text: str, k: int = 3):
    """返回 (chunks, top1_score)。"""
    from sentence_transformers import SentenceTransformer
    q_emb = store.model.encode([query_text]).astype(np.float32)
    faiss.normalize_L2(q_emb)
    scores, indices = store.index.search(q_emb, k)
    chunks = [store.data[i]["content"] for i in indices[0] if i < len(store.data)]
    top1_score = float(scores[0][0]) if len(scores[0]) > 0 else 0.0
    return chunks, top1_score


def is_hit(chunks: list, keywords: list) -> bool:
    for chunk in chunks:
        text = chunk.lower()
        if any(kw.lower() in text for kw in keywords):
            return True
    return False


def main():
    print("=" * 70)
    print("6.2.2 跨语言检索对比实验")
    print("方案A：中文 → 翻译 → 英文编码 → 检索")
    print("方案B：中文 → 直接英文模型编码 → 检索（基线）")
    print("=" * 70)

    store = EmbeddingStore()
    store.load_index(
        str(BASE_DIR / "faiss_index.bin"),
        str(BASE_DIR / "index_data.json"),
    )

    results = []

    print(f"\n{'#':<4} {'中文查询(前15字)':<18} {'翻译结果(前25字)':<28} {'A命中':<6} {'A分数':<8} {'B命中':<6} {'B分数'}")
    print("-" * 80)

    for i, (zh_query, keywords) in enumerate(QUERIES):
        # 方案A：翻译后检索
        en_query = translate_to_english(zh_query)
        time.sleep(0.3)  # 避免API限速
        chunks_a, score_a = search_with_score(store, en_query)
        hit_a = is_hit(chunks_a, keywords)

        # 方案B：直接用英文模型编码中文（基线）
        chunks_b, score_b = search_with_score(store, zh_query)
        hit_b = is_hit(chunks_b, keywords)

        results.append({
            "zh_query": zh_query,
            "en_query": en_query,
            "keywords": keywords,
            "hit_a": hit_a,
            "score_a": score_a,
            "hit_b": hit_b,
            "score_b": score_b,
        })

        mark_a = "✓" if hit_a else "✗"
        mark_b = "✓" if hit_b else "✗"
        print(f"{i+1:<4} {zh_query[:15]:<18} {en_query[:25]:<28} {mark_a:<6} {score_a:<8.4f} {mark_b:<6} {score_b:.4f}")

    # ===== 汇总 =====
    hits_a   = sum(r["hit_a"]   for r in results)
    hits_b   = sum(r["hit_b"]   for r in results)
    avg_score_a = np.mean([r["score_a"] for r in results])
    avg_score_b = np.mean([r["score_b"] for r in results])

    print("\n" + "=" * 70)
    print(f"{'指标':<28} {'方案A（翻译增强）':>18} {'方案B（直接编码）':>18}")
    print("-" * 66)
    print(f"{'Top-3召回命中数（/20）':<28} {hits_a:>18} {hits_b:>18}")
    print(f"{'Top-3召回率':<28} {hits_a/20:>17.1%} {hits_b/20:>17.1%}")
    print(f"{'Top-1平均余弦相似度':<28} {avg_score_a:>18.4f} {avg_score_b:>18.4f}")
    print("=" * 70)

    # 分析未命中案例
    missed_a = [r for r in results if not r["hit_a"]]
    if missed_a:
        print(f"\n方案A未命中的 {len(missed_a)} 条查询：")
        for r in missed_a:
            print(f"  · {r['zh_query']}")
            print(f"    翻译: {r['en_query']}")
            print(f"    期望关键词: {r['keywords']}")


if __name__ == "__main__":
    main()
