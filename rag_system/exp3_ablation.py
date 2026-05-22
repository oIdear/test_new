"""
消融实验：四配置递增对比
C0: all-MiniLM-L6-v2 + 纯向量 + 旧查询词（基线）
C1: bge-base-en-v1.5  + 纯向量 + 旧查询词（换模型）
C2: bge-base-en-v1.5  + 混合检索（BM25+RRF） + 旧查询词（+混合检索）
C3: bge-base-en-v1.5  + 混合检索 + 新语义查询词（完整系统）
"""
import json, sys, time, numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent))
from embedding_store import EmbeddingStore

ALARM_PATH = Path('/home/zzx-king/zzx/test/post_processor/summary_output/alarms_wide_202408.jsonl')
DATA_PATH  = Path('/home/zzx-king/zzx/test/rag_system/parsed_data.json')
OUT_PATH   = Path('/home/zzx-king/zzx/test/experiments/exp3_raw.json')

PATTERN_GROUND_TRUTH = {
    'a1': ['RPKI','ROA','IRR','origin','validation'],
    'a2': ['valley','route leak','provider','customer'],
    'a3': ['reserved','AS_PATH','AS number','autonomous system'],
    'a4': ['origin','autonomous system','organization'],
}

PATTERN_DESCRIPTIONS = {
    'a1': 'origin AS validation RPKI ROA route origin authorization IRR prefix legitimacy',
    'a2': 'valley-free violation route leak provider customer peer AS relationship BGP policy',
    'a3': 'reserved ASN unknown autonomous system bogon path anomaly invalid AS number',
    'a4': 'same organization origin AS change internal routing sibling AS',
    'b1': 'origin connectivity RPKI validation upstream provider customer link',
    'b2': 'AS path prepending traffic engineering load balancing',
    'b3': 'different upstream provider path change origin upstream diversity',
}


def dominant_pattern(alarm):
    counts = {}
    for ev in alarm.get('events', []):
        for rc in ev.get('route_changes', []):
            for k in rc.get('patterns', {}).keys():
                counts[k] = counts.get(k, 0) + 1
    return max(counts, key=counts.get) if counts else 'unknown'


def sample_alarms():
    targets = {'a2': 7, 'a1': 6, 'a3': 5, 'a4': 2}
    buckets = {k: [] for k in targets}
    with open(ALARM_PATH) as f:
        for line in f:
            alarm = json.loads(line)
            dp = dominant_pattern(alarm)
            if dp in buckets and len(buckets[dp]) < targets[dp]:
                buckets[dp].append(alarm)
    return [a for v in buckets.values() for a in v]


def build_old_query(alarm):
    parts = ['BGP routing anomaly detection']
    for ev in alarm.get('events', []):
        for rc in ev.get('route_changes', []):
            if rc.get('path1'): parts.append('path before: ' + str(rc['path1'])[:80])
            if rc.get('path2'): parts.append('path after: ' + str(rc['path2'])[:80])
            diff = rc.get('diff')
            if diff is not None and diff != float('inf'): parts.append('beam_diff_score: ' + str(round(diff, 4)))
            elif diff == float('inf'): parts.append('beam_diff_score: infinity')
            patterns = rc.get('patterns', {})
            if 'a2' in patterns: parts.append('valley_free_violation detected')
            a1 = patterns.get('a1', {})
            for k in ['origin_rpki_1', 'origin_rpki_2', 'origin_irr_1', 'origin_irr_2']:
                if a1.get(k): parts.append(k + ': ' + a1[k])
    parts.append('route leak hijack misconfiguration RPKI IRR ROA valley-free BGP security')
    return ' '.join(parts)[:2000]


def build_new_query(alarm):
    parts = ['BGP routing anomaly']
    seen = set()
    for ev in alarm.get('events', []):
        for rc in ev.get('route_changes', []):
            for pk in rc.get('patterns', {}).keys():
                seen.add(pk)
    for pk in sorted(seen):
        if pk in PATTERN_DESCRIPTIONS:
            parts.append(PATTERN_DESCRIPTIONS[pk])
    for ev in alarm.get('events', []):
        for rc in ev.get('route_changes', []):
            a1 = rc.get('patterns', {}).get('a1', {})
            for field in ['origin_rpki_1', 'origin_rpki_2']:
                val = a1.get(field, '')
                if val:
                    parts.append('RPKI status ' + val)
                    break
    if any(rc.get('diff') == float('inf') for ev in alarm.get('events', []) for rc in ev.get('route_changes', [])):
        parts.append('extreme path anomaly unseen AS relationship')
    return ' '.join(parts)[:500]


def build_temp_store(model_name):
    """用指定模型名构建临时内存索引（不保存到磁盘）。"""
    print(f'  建立临时索引 model={model_name}...', flush=True)
    t0 = time.time()
    model = SentenceTransformer(model_name)
    with open(DATA_PATH) as f:
        data = json.load(f)
    texts = [item['content'] for item in data]
    # 添加 bge 前缀（仅 doc 侧不需要，query 侧需要）
    embs = model.encode(texts, show_progress_bar=True, batch_size=32).astype(np.float32)
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    print(f'  完成，耗时 {time.time()-t0:.1f}s', flush=True)
    return model, index, data, bm25


def vector_search(model, index, data, query, k=3, bge=False):
    prefix = 'Represent this sentence for searching relevant passages: ' if bge else ''
    emb = model.encode([prefix + query]).astype(np.float32)
    faiss.normalize_L2(emb)
    dists, idxs = index.search(emb, k)
    return [(data[i], float(dists[0][j])) for j, i in enumerate(idxs[0]) if i < len(data)]


def hybrid_search(model, index, data, bm25, query, k=3, bge=False, mmr=True):
    # 向量检索
    prefix = 'Represent this sentence for searching relevant passages: ' if bge else ''
    emb = model.encode([prefix + query]).astype(np.float32)
    faiss.normalize_L2(emb)
    K = min(20, len(data))
    dists, idxs = index.search(emb, K)
    vec_res = [(data[i], float(dists[0][j])) for j, i in enumerate(idxs[0]) if i < len(data)]

    # BM25
    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25 = np.argsort(bm25_scores)[::-1][:K]
    bm25_res = [(data[i], float(bm25_scores[i])) for i in top_bm25 if bm25_scores[i] > 0]

    # RRF
    c = 60
    scores = {}
    for rank, (item, _) in enumerate(vec_res):
        key = item['content'][:50]
        scores[key] = scores.get(key, {'item': item, 'score': 0.0})
        scores[key]['score'] += 1.0 / (c + rank + 1)
    for rank, (item, _) in enumerate(bm25_res):
        key = item['content'][:50]
        scores[key] = scores.get(key, {'item': item, 'score': 0.0})
        scores[key]['score'] += 1.0 / (c + rank + 1)
    ranked = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
    merged = [(r['item'], r['score']) for r in ranked[:k * 3]]

    if not mmr or len(merged) <= k:
        return merged[:k]

    # MMR
    selected, candidates = [], list(merged)
    while len(selected) < k and candidates:
        if not selected:
            selected.append(candidates.pop(0))
            continue
        best = max(range(len(candidates)), key=lambda i: (
            0.6 * candidates[i][1] - 0.4 * max(
                len(set(candidates[i][0]['content'].lower().split()) & set(s[0]['content'].lower().split())) /
                max(len(set(candidates[i][0]['content'].lower().split()) | set(s[0]['content'].lower().split())), 1)
                for s in selected
            )
        ))
        selected.append(candidates.pop(best))
    return selected


def hit(results, kws):
    return any(any(k.lower() in r[0]['content'].lower() for k in kws) for r in results)


def diversity(results):
    return len(set(r[0].get('file', '?') for r in results))


def main():
    alarms = sample_alarms()
    print(f'Sampled {len(alarms)} alarms', flush=True)

    # ===== C0: all-MiniLM + 纯向量 + 旧查询 =====
    print('\n=== C0: all-MiniLM + pure vector + old query ===', flush=True)
    m0, idx0, data0, bm25_0 = build_temp_store('all-MiniLM-L6-v2')

    # ===== C1/C2/C3: bge + 新索引（已有） =====
    print('\n=== 加载 bge 索引（C1/C2/C3 共用） ===', flush=True)
    store_bge = EmbeddingStore()
    store_bge.load_index('faiss_index.bin', 'index_data.json')
    m1, idx1, data1, bm25_1 = store_bge.model, store_bge.index, store_bge.data, store_bge.bm25

    rows = []
    for alarm in alarms:
        dp = dominant_pattern(alarm)
        gid = alarm.get('group_id', '?')
        kws = PATTERN_GROUND_TRUTH.get(dp, ['BGP'])
        q_old = build_old_query(alarm)
        q_new = build_new_query(alarm)

        # C0
        r0 = vector_search(m0, idx0, data0, q_old, k=3, bge=False)
        # C1
        r1 = vector_search(m1, idx1, data1, q_old, k=3, bge=True)
        # C2
        r2 = hybrid_search(m1, idx1, data1, bm25_1, q_old, k=3, bge=True, mmr=False)
        # C3
        r3 = hybrid_search(m1, idx1, data1, bm25_1, q_new, k=3, bge=True, mmr=True)

        row = {
            'gid': gid, 'dp': dp, 'kws': kws,
            'C0_hit': hit(r0, kws), 'C0_div': diversity(r0),
            'C1_hit': hit(r1, kws), 'C1_div': diversity(r1),
            'C2_hit': hit(r2, kws), 'C2_div': diversity(r2),
            'C3_hit': hit(r3, kws), 'C3_div': diversity(r3),
        }
        rows.append(row)
        print(f'  gid={gid} dp={dp} C0={row["C0_hit"]} C1={row["C1_hit"]} C2={row["C2_hit"]} C3={row["C3_hit"]}', flush=True)

    with open(OUT_PATH, 'w') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f'\nSaved to {OUT_PATH}')

    # 输出摘要
    for cfg in ['C0', 'C1', 'C2', 'C3']:
        total = len(rows)
        h = sum(r[cfg + '_hit'] for r in rows)
        d = sum(r[cfg + '_div'] for r in rows) / total
        print(f'{cfg}: recall={h}/{total}({h/total:.0%}) div={d:.2f}')


if __name__ == '__main__':
    main()
