#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能分块算法对比测试
实验文档：RFC 4271（BGP-4协议规范）
对比方案：固定长度切分 vs RecursiveCharacterTextSplitter递归切分
评估维度：块数量、跨句切割率、Top-3检索召回率
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Tuple
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

# RFC 4271核心概念测试查询集（20条），每条附带验证关键词
TEST_QUERIES: List[Tuple[str, List[str]]] = [
    ("BGP OPEN message format version number",              ["OPEN", "version"]),
    ("AS_PATH attribute AS_SEQUENCE AS_SET segments",       ["AS_PATH", "AS_SEQUENCE"]),
    ("BGP UPDATE message withdrawn routes NLRI",            ["UPDATE", "Withdrawn"]),
    ("NOTIFICATION message error code subcode",             ["NOTIFICATION", "Error"]),
    ("KEEPALIVE message format length",                     ["KEEPALIVE"]),
    ("BGP finite state machine Idle Connect Active",        ["Idle", "Active"]),
    ("OpenSent OpenConfirm Established BGP states",         ["OpenSent", "Established"]),
    ("NEXT_HOP path attribute IP address",                  ["NEXT_HOP"]),
    ("ORIGIN attribute IGP EGP INCOMPLETE",                 ["ORIGIN", "IGP"]),
    ("LOCAL_PREF local preference attribute",               ["LOCAL_PREF"]),
    ("MULTI_EXIT_DISC discriminator selection",             ["MULTI_EXIT_DISC"]),
    ("ATOMIC_AGGREGATE path attribute",                     ["ATOMIC_AGGREGATE"]),
    ("BGP decision process route selection degree of preference", ["Decision", "preference"]),
    ("BGP hold time negotiation timer value",               ["Hold Time"]),
    ("BGP message header marker length type field",         ["marker", "Length", "Type"]),
    ("BGP TCP connection collision detection",              ["collision"]),
    ("route aggregation prefix announcement",               ["aggregat"]),
    ("path attribute flags optional transitive well-known", ["optional", "transitive"]),
    ("Minimum Route Advertisement Interval",                ["Minimum Route Advertisement"]),
    ("BGP Autonomous System speaker BGP Identifier",        ["BGP Identifier", "Autonomous System"]),
]


def load_rfc4271_text() -> str:
    base_dir = Path(__file__).resolve().parent.parent
    pdf_path = base_dir / "bgp-embedding-data" / "rfc4271.txt.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到文件: {pdf_path}")
    print(f"正在读取: {pdf_path.name}")
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)
    print(f"提取文本长度: {len(text)} 字符，共 {len(reader.pages)} 页")
    return text


def fixed_length_split(text: str) -> List[str]:
    chunks, start = [], 0
    step = CHUNK_SIZE - CHUNK_OVERLAP
    while start < len(text):
        chunk = text[start: start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def recursive_split(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def mid_sentence_cut_rate(chunks: List[str]) -> Tuple[int, float]:
    """
    判断跨句切割：块末尾不以句终标点结尾则认为是跨句切割。
    返回 (跨句切割块数, 比例)
    """
    sentence_endings = {".", "!", "?", ":", ";"}
    cuts = sum(1 for c in chunks if c.strip() and c.strip()[-1] not in sentence_endings)
    return cuts, cuts / len(chunks) if chunks else 0.0


def build_faiss_index(chunks: List[str], model: SentenceTransformer):
    embeddings = model.encode(chunks, show_progress_bar=False, batch_size=64)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def top3_recall(chunks: List[str], index, model: SentenceTransformer) -> Tuple[int, float]:
    """
    对每条测试查询检索Top-3块，若任一块包含所有验证关键词则命中。
    返回 (命中查询数, 召回率)
    """
    hits = 0
    for query_text, keywords in TEST_QUERIES:
        q_emb = model.encode([query_text]).astype(np.float32)
        faiss.normalize_L2(q_emb)
        _, indices = index.search(q_emb, TOP_K)
        retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]
        # 任意一个块包含所有关键词（大小写不敏感）即命中
        if any(
            all(kw.lower() in chunk.lower() for kw in keywords)
            for chunk in retrieved
        ):
            hits += 1
    return hits, hits / len(TEST_QUERIES)


def main():
    print("=" * 70)
    print("智能分块算法对比实验  —  RFC 4271专项验证")
    print(f"配置：chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, Top-K={TOP_K}")
    print("=" * 70)

    text = load_rfc4271_text()

    print("\n[1/4] 执行固定长度切分...")
    fixed_chunks = fixed_length_split(text)
    print(f"[2/4] 执行递归字符切分...")
    recursive_chunks = recursive_split(text)

    fixed_cuts, fixed_cut_rate = mid_sentence_cut_rate(fixed_chunks)
    rec_cuts, rec_cut_rate = mid_sentence_cut_rate(recursive_chunks)

    print(f"[3/4] 加载Embedding模型：{MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[4/4] 构建FAISS索引并评估Top-{TOP_K}检索召回率...")
    fixed_index = build_faiss_index(fixed_chunks, model)
    rec_index   = build_faiss_index(recursive_chunks, model)

    fixed_hits, fixed_recall = top3_recall(fixed_chunks, fixed_index, model)
    rec_hits,   rec_recall   = top3_recall(recursive_chunks, rec_index, model)

    # ===== 输出结果 =====
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    header = f"{'指标':<28} {'固定长度切分':>14} {'递归字符切分':>14}"
    print(header)
    print("-" * 60)
    print(f"{'总块数':<28} {len(fixed_chunks):>14} {len(recursive_chunks):>14}")
    print(f"{'平均块大小（字符）':<28} {sum(len(c) for c in fixed_chunks)//len(fixed_chunks):>14} {sum(len(c) for c in recursive_chunks)//len(recursive_chunks):>14}")
    print(f"{'跨句切割块数':<28} {fixed_cuts:>14} {rec_cuts:>14}")
    print(f"{'跨句切割率':<28} {fixed_cut_rate:>13.1%} {rec_cut_rate:>13.1%}")
    print(f"{'Top-3召回命中查询数（/20）':<28} {fixed_hits:>14} {rec_hits:>14}")
    print(f"{'Top-3检索召回率':<28} {fixed_recall:>13.1%} {rec_recall:>13.1%}")
    print("=" * 70)

    # ===== 逐条查询明细 =====
    print("\n各查询命中明细：")
    print(f"  {'查询':<50} {'固定':>6} {'递归':>6}")
    print("  " + "-" * 65)

    q_embs = model.encode([q for q, _ in TEST_QUERIES]).astype(np.float32)
    faiss.normalize_L2(q_embs)
    _, fixed_all_idx = fixed_index.search(q_embs, TOP_K)
    _, rec_all_idx   = rec_index.search(q_embs, TOP_K)

    for i, (query_text, keywords) in enumerate(TEST_QUERIES):
        f_retrieved = [fixed_chunks[j] for j in fixed_all_idx[i] if j < len(fixed_chunks)]
        r_retrieved = [recursive_chunks[j] for j in rec_all_idx[i] if j < len(recursive_chunks)]
        f_hit = "✓" if any(all(kw.lower() in c.lower() for kw in keywords) for c in f_retrieved) else "✗"
        r_hit = "✓" if any(all(kw.lower() in c.lower() for kw in keywords) for c in r_retrieved) else "✗"
        print(f"  {query_text[:48]:<50} {f_hit:>6} {r_hit:>6}")


if __name__ == "__main__":
    main()
