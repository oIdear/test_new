import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from embedding_store import EmbeddingStore

store = EmbeddingStore()
store.load_index('faiss_index.bin', 'index_data.json')

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

alarm_path = Path('/home/zzx-king/zzx/test/post_processor/summary_output/alarms_wide_202408.jsonl')
targets = {'a2': 7, 'a1': 6, 'a3': 5, 'a4': 2}
buckets = {k: [] for k in targets}
with open(alarm_path) as f:
    for line in f:
        alarm = json.loads(line)
        dp = dominant_pattern(alarm)
        if dp in buckets and len(buckets[dp]) < targets[dp]:
            buckets[dp].append(alarm)

selected = [a for v in buckets.values() for a in v]
print(f'Selected {len(selected)} alarms')

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
    parts.append('route leak hijack misconfiguration RPKI IRR ROA valley-free BGP security')
    return ' '.join(parts)[:2000]

def hit(results, kws):
    return any(any(k.lower() in r[0]['content'].lower() for k in kws) for r in results)

rows = []
for alarm in selected:
    dp = dominant_pattern(alarm)
    gid = alarm.get('group_id', '?')
    kws = PATTERN_GROUND_TRUTH.get(dp, ['BGP'])
    qn = build_new_query(alarm)
    qo = build_old_query(alarm)
    rn = store.search(qn, k=3, hybrid=True)
    ro = store.search(qo, k=3, hybrid=False)
    files_n = [r[0].get('file', '?') for r in rn]
    files_o = [r[0].get('file', '?') for r in ro]
    row = {
        'gid': gid, 'dp': dp, 'kws': kws,
        'hit_new': hit(rn, kws), 'div_new': len(set(files_n)), 'files_new': files_n,
        'hit_old': hit(ro, kws), 'div_old': len(set(files_o)), 'files_old': files_o,
        'qlen_new': len(qn), 'qlen_old': len(qo),
    }
    rows.append(row)
    print(f'  gid={gid} dp={dp} hit_new={row["hit_new"]} hit_old={row["hit_old"]}')

out = Path('/home/zzx-king/zzx/test/experiments/exp2_raw.json')
with open(out, 'w') as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f'\nSaved {len(rows)} rows to {out}')
