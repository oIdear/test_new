#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6.3.1 RAG增强 vs 无RAG 对比实验
选取 group_id=0 的真实告警，分别在有/无RAG上下文条件下调用 DeepSeek，
捕获两段真实 LLM 输出用于论文案例对比。
"""

import sys
import json
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from embedding_store import EmbeddingStore

DEEPSEEK_API_KEY = "sk-18be0971348e40b6b5ff9ffcfe19a63e"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
TARGET_GROUP_ID  = 8


def load_alarm(group_id: int) -> dict:
    path = BASE_DIR.parent / "post_processor" / "summary_output" / "alarms_wide_202408.jsonl"
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj["group_id"] == group_id:
                return obj
    raise ValueError(f"group_id={group_id} 不存在")


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
            culprit_list = rc.get("culprit", [])
            culprit_asns = [asn for grp in culprit_list if isinstance(grp, list) for asn in grp]
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
                if "none_rel_1" in a3 or "none_rel_2" in a3:
                    parts.append("no_business_relationship_in_path")
            if "b1" in patterns:
                conn = patterns["b1"].get("origin_connection", "")
                if conn:
                    parts.append(f"origin_connection_type: {conn}")
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
    ctx = "\n\n".join(blocks)
    print(f"\n--- RAG检索上下文（查询词前100字）---")
    print(query[:100], "...")
    print(f"--- 检索到 {len(results)} 个相关文档块 ---")
    for item, dist in results:
        print(f"  [{item.get('file','?')} | 相似度:{dist:.3f}] {item['content'][:80]}...")
    return ctx


ANALYSIS_PROMPT_TEMPLATE = """
你是一名BGP异常检测与溯源分析专家，同时具备RFC标准、AS关系模型和路由策略分析经验。

现在提供两类信息：
一类是【异常观测数据】（来自BGP检测系统的结构化事件）
一类是【检索增强上下文】（来自RFC文档、研究论文和运维知识库的相关资料）

请基于检索证据进行约束推理，禁止脱离证据臆测结论。
分析目标是：对异常进行溯源解释与归因判断。

====================
【异常观测数据】
{alarm_json}
====================

{rag_section}

请输出一份结构化溯源分析报告，必须包含：

【1】异常类型判定
- 判定属于：路由泄露 / 路由劫持 / 错误宣告 / 策略异常 / 可疑但不确定
- 给出判定依据

【2】异常传播与路径特征分析
- AS路径变化模式
- 是否出现 valley-free 破坏 / 非法上游传播 / 路径突变

【3】关键证据链
- 用"证据 → 推论"形式说明
- 尽量引用RFC或研究资料中的判据依据

【4】验证建议（运维可执行）
- 建议检查：RPKI / IRR / AS关系 / 上游策略

【5】置信度评分
- 给出 0–1 之间置信度，说明不确定来源
"""


def call_deepseek(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "你是一个BGP异常溯源分析专家，基于提供的上下文信息，对BGP异常进行分析和解释。"},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }
    resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    print("=" * 70)
    print(f"6.3.1 RAG对比实验  —  group_id={TARGET_GROUP_ID}")
    print("=" * 70)

    alarm = load_alarm(TARGET_GROUP_ID)
    print(f"告警加载成功：前缀={alarm['events'][0]['prefix'][0]}，"
          f"时间={alarm['start_time']}")

    # ===== 加载向量索引 =====
    store = EmbeddingStore()
    store.load_index(
        str(BASE_DIR / "faiss_index.bin"),
        str(BASE_DIR / "index_data.json"),
    )

    # ===== 构建RAG上下文 =====
    rag_context = build_rag_context(alarm, store, k=3)

    alarm_json = json.dumps(alarm, indent=2, ensure_ascii=False)

    # ===== 无RAG调用 =====
    print("\n" + "=" * 70)
    print("【调用1/2】无RAG配置（仅提供异常数据，不注入检索资料）")
    print("=" * 70)
    prompt_no_rag = ANALYSIS_PROMPT_TEMPLATE.format(
        alarm_json=alarm_json,
        rag_section="【检索增强上下文】\n（本次未启用RAG检索，无参考资料注入）"
    )
    response_no_rag = call_deepseek(prompt_no_rag)
    print(response_no_rag)

    # ===== 有RAG调用 =====
    print("\n" + "=" * 70)
    print("【调用2/2】有RAG配置（注入FAISS检索到的RFC文档片段）")
    print("=" * 70)
    prompt_with_rag = ANALYSIS_PROMPT_TEMPLATE.format(
        alarm_json=alarm_json,
        rag_section=f"【检索增强上下文】\n{rag_context}"
    )
    response_with_rag = call_deepseek(prompt_with_rag)
    print(response_with_rag)

    # ===== 保存结果 =====
    output_path = BASE_DIR / "rag_comparison_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "group_id": TARGET_GROUP_ID,
            "alarm_summary": {
                "prefix": alarm["events"][0]["prefix"][0],
                "start_time": alarm["start_time"],
            },
            "rag_context_sources": [
                {"file": item.get("file"), "score": float(dist)}
                for item, dist in store.search(build_query(alarm), k=3)
            ],
            "response_no_rag":   response_no_rag,
            "response_with_rag": response_with_rag,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
