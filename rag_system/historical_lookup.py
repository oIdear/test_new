"""
历史异常案例库精确匹配模块

从告警中提取 ASN 和前缀，对三类历史 CSV 做结构化查找：
  - Prefix-Anomaly.csv   : 前缀劫持案例（attacker/victim AS + 前缀）
  - Path-Anomaly.csv     : 路径异常案例（affected_asn/suspicious_links + 前缀）
  - Route-Leak-Anomaly.csv: 路由泄露案例（affected_asn/leak_asn + 前缀）

与向量检索互补：向量检索负责召回规范文档，本模块负责命中私有历史案例。
"""

import csv
import ipaddress
import re
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / 'bgp-embedding-data'

# ------------------------------------------------------------------ #
#  启动时加载，全局缓存
# ------------------------------------------------------------------ #
_prefix_cases = []
_path_cases = []
_leak_cases = []
_loaded = False


def _load():
    global _prefix_cases, _path_cases, _leak_cases, _loaded
    if _loaded:
        return

    def read(fname):
        path = _DATA_DIR / fname
        if not path.exists():
            return []
        with open(path, encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))

    _prefix_cases = read('Prefix-Anomaly.csv')
    _path_cases   = read('Path-Anomaly.csv')
    _leak_cases   = read('Route-Leak-Anomaly.csv')
    _loaded = True
    print(f'历史案例库加载完成：'
          f'前缀劫持 {len(_prefix_cases)} 条，'
          f'路径异常 {len(_path_cases)} 条，'
          f'路由泄露 {len(_leak_cases)} 条')


# ------------------------------------------------------------------ #
#  前缀匹配工具
# ------------------------------------------------------------------ #
def _parse_network(s):
    """将前缀字符串解析为 IPv4Network，失败返回 None。"""
    s = s.strip()
    # 去除 HTML 标签（Prefix-Anomaly 的 example_prefix 字段含 <div>）
    s = re.sub(r'<[^>]+>', '', s).strip()
    # 取第一个看起来像 CIDR 的 token
    for tok in s.replace(';', ' ').split():
        try:
            return ipaddress.ip_network(tok, strict=False)
        except ValueError:
            continue
    return None


def _prefixes_overlap(net_a, net_b):
    """两个前缀是否有包含关系（完全重叠 / 子前缀 / 超前缀）。"""
    if net_a is None or net_b is None:
        return False
    return net_a.overlaps(net_b)


def _expand_prefixes(raw_str):
    """解析分号/逗号分隔的多前缀字符串，返回 IPv4Network 列表。"""
    nets = []
    for tok in re.split(r'[;,\s]+', raw_str):
        n = _parse_network(tok)
        if n:
            nets.append(n)
    return nets


# ------------------------------------------------------------------ #
#  告警信息提取
# ------------------------------------------------------------------ #
def _extract_alarm_info(alarm):
    """从告警 JSON 中提取所有 ASN 和宣告前缀。"""
    asns = set()
    prefixes = []

    for event in alarm.get('events', []):
        # 宣告前缀（可能是字符串或列表）
        raw_prefix = event.get('prefix', '')
        if isinstance(raw_prefix, list):
            raw_prefix = raw_prefix[0] if raw_prefix else ''
        if raw_prefix:
            net = _parse_network(str(raw_prefix))
            if net:
                prefixes.append(net)

        for rc in event.get('route_changes', []):
            # path1 / path2 是空格分隔的 ASN 字符串
            for path_key in ('path1', 'path2'):
                for tok in str(rc.get(path_key, '') or '').split():
                    try:
                        asns.add(int(tok))
                    except ValueError:
                        pass
            # culprit 是 [[asn, asn, ...], ...] 结构
            for group in (rc.get('culprit') or []):
                for tok in (group if isinstance(group, list) else [group]):
                    try:
                        asns.add(int(tok))
                    except (ValueError, TypeError):
                        pass
            # patterns 中可能有 origin_as
            for pat_val in rc.get('patterns', {}).values():
                if isinstance(pat_val, dict):
                    for field in ('origin_as_1', 'origin_as_2'):
                        try:
                            asns.add(int(pat_val[field]))
                        except (KeyError, ValueError, TypeError):
                            pass

    return asns, prefixes


# ------------------------------------------------------------------ #
#  三类案例库查找
# ------------------------------------------------------------------ #
def _search_prefix_anomaly(alarm_asns, alarm_prefixes, max_hits=3):
    hits = []
    for row in _prefix_cases:
        try:
            attacker = int(row.get('attacker', '') or 0)
            victim   = int(row.get('victim',   '') or 0)
        except ValueError:
            continue

        as_match = (attacker in alarm_asns) or (victim in alarm_asns)

        prefix_match = False
        for prefix_str in (row.get('prefix_list', '') + ';' + row.get('example_prefix', '')).split(';'):
            hist_net = _parse_network(prefix_str)
            if any(_prefixes_overlap(hist_net, ap) for ap in alarm_prefixes):
                prefix_match = True
                break

        if as_match or prefix_match:
            match_basis = []
            if as_match:
                match_basis.append(f'AS {attacker}（攻击者）' if attacker in alarm_asns
                                   else f'AS {victim}（受害者）')
            if prefix_match:
                match_basis.append('前缀重叠')

            hits.append({
                'source': 'Prefix-Anomaly.csv',
                'match_basis': '、'.join(match_basis),
                'event_type': row.get('event_type', ''),
                'attacker_as': attacker,
                'victim_as': victim,
                'level': row.get('level', ''),
                'prefixes': row.get('prefix_list', '').replace(';', ', '),
                'start_time': row.get('start_time', ''),
            })
            if len(hits) >= max_hits:
                break
    return hits


def _search_path_anomaly(alarm_asns, alarm_prefixes, max_hits=3):
    hits = []
    for row in _path_cases:
        try:
            affected_asn = int(row.get('affected_asn', '') or 0)
        except ValueError:
            affected_asn = 0

        suspicious_asns = set()
        for tok in re.split(r'[;,\s]+', row.get('suspicious_links', '')):
            try:
                suspicious_asns.add(int(tok))
            except ValueError:
                pass

        involved_asns = ({affected_asn} | suspicious_asns) - {0}
        as_match = bool(involved_asns & alarm_asns)

        prefix_match = False
        hist_nets = _expand_prefixes(row.get('affected_prefix', '') + ';' +
                                     row.get('prefix_list', ''))
        for hn in hist_nets:
            if any(_prefixes_overlap(hn, ap) for ap in alarm_prefixes):
                prefix_match = True
                break

        if as_match or prefix_match:
            match_basis = []
            matched_as = involved_asns & alarm_asns
            if matched_as:
                match_basis.append(f'AS {list(matched_as)[0]} 曾出现在异常路径中')
            if prefix_match:
                match_basis.append('前缀重叠')

            hits.append({
                'source': 'Path-Anomaly.csv',
                'match_basis': '、'.join(match_basis),
                'event_type': row.get('event_type', ''),
                'affected_asn': affected_asn,
                'affected_prefix': row.get('affected_prefix', ''),
                'suspicious_links': row.get('suspicious_links', ''),
                'status': row.get('status', ''),
                'start_time': row.get('start_time', ''),
            })
            if len(hits) >= max_hits:
                break
    return hits


def _search_leak_anomaly(alarm_asns, alarm_prefixes, max_hits=3):
    hits = []
    for row in _leak_cases:
        try:
            affected_asn = int(row.get('affected_asn', '') or 0)
            leak_asn     = int(row.get('leak_asn',     '') or 0)
        except ValueError:
            affected_asn = leak_asn = 0

        involved_asns = {affected_asn, leak_asn} - {0}
        as_match = bool(involved_asns & alarm_asns)

        prefix_match = False
        hist_nets = _expand_prefixes(row.get('affected_prefix', '') + ';' +
                                     row.get('prefix_list', ''))
        for hn in hist_nets:
            if any(_prefixes_overlap(hn, ap) for ap in alarm_prefixes):
                prefix_match = True
                break

        if as_match or prefix_match:
            match_basis = []
            if leak_asn in alarm_asns:
                match_basis.append(f'AS {leak_asn} 曾是路由泄露发起者')
            elif affected_asn in alarm_asns:
                match_basis.append(f'AS {affected_asn} 曾是路由泄露受害者')
            if prefix_match:
                match_basis.append('前缀重叠')

            hits.append({
                'source': 'Route-Leak-Anomaly.csv',
                'match_basis': '、'.join(match_basis),
                'event_type': row.get('event_type', ''),
                'affected_asn': affected_asn,
                'leak_asn': leak_asn,
                'affected_prefix': row.get('affected_prefix', ''),
                'status': row.get('status', ''),
                'start_time': row.get('start_time', ''),
            })
            if len(hits) >= max_hits:
                break
    return hits


# ------------------------------------------------------------------ #
#  公开接口
# ------------------------------------------------------------------ #
def search_historical_cases(alarm, max_per_type=2):
    """
    对告警做历史案例精确匹配，返回格式化的上下文字符串。
    无命中时返回空字符串。
    """
    _load()
    alarm_asns, alarm_prefixes = _extract_alarm_info(alarm)

    if not alarm_asns and not alarm_prefixes:
        return ''

    prefix_hits = _search_prefix_anomaly(alarm_asns, alarm_prefixes, max_per_type)
    path_hits   = _search_path_anomaly(alarm_asns, alarm_prefixes, max_per_type)
    leak_hits   = _search_leak_anomaly(alarm_asns, alarm_prefixes, max_per_type)

    all_hits = prefix_hits + path_hits + leak_hits
    if not all_hits:
        return ''

    blocks = []
    for h in all_hits:
        src = h['source']
        basis = h['match_basis']
        etype = h['event_type']

        if src == 'Prefix-Anomaly.csv':
            detail = (f"事件类型：{etype}；"
                      f"攻击者 AS {h['attacker_as']}，受害者 AS {h['victim_as']}；"
                      f"涉及前缀：{h['prefixes']}；"
                      f"严重程度：{h['level']}")
        elif src == 'Path-Anomaly.csv':
            detail = (f"事件类型：{etype}；"
                      f"受影响 AS {h['affected_asn']}，前缀 {h['affected_prefix']}；"
                      f"可疑路径 AS 对：{h['suspicious_links']}；"
                      f"状态：{h['status']}")
        else:  # Route-Leak
            detail = (f"事件类型：{etype}；"
                      f"泄露发起 AS {h['leak_asn']}，受害 AS {h['affected_asn']}；"
                      f"前缀 {h['affected_prefix']}；"
                      f"状态：{h['status']}")

        blocks.append(f"[历史案例来源:{src} | 匹配依据:{basis}]\n{detail}")

    return '\n\n'.join(blocks)


if __name__ == '__main__':
    # 快速验证
    import json
    from pathlib import Path as P
    alarm_path = P(__file__).resolve().parent.parent / 'post_processor/summary_output/alarms_wide_202408.jsonl'
    with open(alarm_path) as f:
        alarm = json.loads(f.readline())
    result = search_historical_cases(alarm)
    print(f'告警 gid={alarm["group_id"]} 历史匹配结果：')
    print(result if result else '（无匹配）')
