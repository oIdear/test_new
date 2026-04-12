#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
脚本功能：从Cisco BGPStream官网爬取BGP异常事件数据（如劫持、泄露、断连等），并缓存到本地
核心流程： 
1. 爬取BGPStream首页的事件列表
2. 解析不同类型异常事件的关键信息（ASN、前缀、国家等）
3. 增量更新本地缓存（仅获取未缓存的新事件）
4. 补充事件详情页的关键字段（如劫持的预期/检测前缀、泄露的前缀等）
"""

from pathlib import Path
import numpy as np
from urllib.parse import urljoin  # 用于拼接URL
from concurrent.futures import ThreadPoolExecutor  # 多线程加速详情页爬取
import subprocess  # 调用curl命令爬取页面
import json  # 存储事件数据为JSON格式
import re  # 正则解析HTML内容

# ========== 路径配置 ==========
# 脚本所在目录
SCRIPT_DIR = Path(__file__).resolve().parent
# 缓存目录（存储爬取的事件数据）
CACHE_DIR = SCRIPT_DIR/"cache"
# 确保缓存目录存在，不存在则创建
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# BGPStream官网地址（数据源）
url_index="https://bgpstream.crosswork.cisco.com/"

def get_page(url):
    """
    爬取指定URL的页面内容
    :param url: 目标URL
    :return: 页面的字符串内容（UTF-8编码）
    """
    # 调用curl命令静默爬取页面（-s参数：不输出进度/错误信息）
    page = subprocess.check_output(["curl", "-s", url]).decode()
    return page

def item_parser(index_page):
    """
    解析BGPStream首页的事件列表，提取各类BGP异常事件的核心信息
    :param index_page: BGPStream首页的HTML内容
    :return: 
        events: 解析后的事件列表（每个元素是字典，包含事件类型、ASN、国家等）
        ids: 事件ID列表（用于排序和增量更新）
    """
    events = []
    # 正则匹配HTML中的<tr>行（每个<tr>对应一个事件），re.DOTALL使.匹配换行符
    for item_str in re.finditer(r'\<tr\>.+?\</tr\>', index_page, flags=re.DOTALL):
        try:
            item_str = item_str[0]  # 获取匹配到的<tr>内容
            item = dict()  # 存储单个事件的解析结果
            # 匹配<tr>内的<td>标签，提取class属性（字段名）和标签内容（字段值）
            for k, v in re.findall(r'\<td class="(.+?)"\>(.+?)\</td\>',
                    item_str, flags=re.DOTALL):
                # 清理字段值：替换多空格为单空格、移除换行符、首尾去空格
                v = re.sub(r"\s\s+", " ", v.replace("\n", " ")).strip()

                # ========== 按字段名解析不同类型的事件 ==========
                if k == "asn":
                    # 提取ASN（格式如"(AS 1234)"）
                    asns = re.findall(r'\(AS (\d+?)\)', v, flags=re.DOTALL)

                    # 不同事件类型的ASN解析逻辑不同
                    if item["event_type"] == "Outage":
                        # 断连事件：可能包含多个ASN
                        item["asn"] = asns
                    elif item["event_type"] == "Possible Hijack":
                        # 疑似劫持事件：分为预期ASN和检测到的ASN
                        expected, detected = asns
                        item["expected_asn"] = expected
                        item["detected_asn"] = detected
                    elif item["event_type"] == "BGP Leak":
                        # BGP泄露事件：分为源ASN和泄露者ASN
                        origin, leaker = asns
                        item["origin_asn"] = origin
                        item["leaker_asn"] = leaker
                    else:
                        # 未覆盖的事件类型，抛出异常
                        raise RuntimeError(f"Uncovered event_type: {item['event_type']}")
                
                elif k == "country" and v:
                    # 解析国家（取第一个字段，如"US United States"取"US"）
                    item["country"] = v.split(" ")[0]
                
                elif k == "moredetail":
                    # 解析详情页链接和事件ID
                    v = re.search(r'\<a href="(.+?(\d+))"\>', v)
                    if v is None:
                        # 无详情页链接的情况
                        item["moredetail"] = ""
                        item["event_id"] = ""
                    else:
                        # 提取详情页URL和事件ID（ID是URL最后数字）
                        item["moredetail"] = v[1]
                        item["event_id"] = v[2]
                
                else:
                    # 其他字段（如event_type、start_time、end_time等）直接赋值
                    item[k] = v
            
            # 将解析后的单个事件加入列表
            events.append(item)
        
        except Exception as e:
            # 单个事件解析失败时打印异常，继续处理下一个事件
            print(e)
            continue
    
    # ========== 事件排序 ==========
    # 提取所有事件ID并转为整数
    ids = np.array([int(i["event_id"]) for i in events])
    # 按事件ID升序排序的索引
    sort_idx = np.argsort(ids)
    # 按索引重新排序事件和ID
    events = np.array(events)[sort_idx].tolist()
    ids = ids[sort_idx].tolist()
    
    return events, ids

def update_cache():
    """
    增量更新本地缓存：仅爬取未缓存的新事件，并补充详情页信息
    """
    # 1. 爬取BGPStream首页（事件列表页）
    index_page = get_page(url_index)
    # 2. 解析首页，得到事件列表和事件ID列表
    events, ids = item_parser(index_page)

    # 3. 确定需要增量更新的事件范围
    # 获取本地已缓存的事件ID（缓存文件命名为"事件ID.jsonl"）
    current_id = [int(i.stem) for i in CACHE_DIR.glob("*.jsonl")]
    # 本地缓存的最大事件ID（无缓存则为-1）
    current_max_id = max(current_id) if current_id else -1
    # 找到第一个大于current_max_id的事件索引（即新事件的起始位置）
    start_idx = np.searchsorted(ids, current_max_id, "right")

    # 筛选出未缓存的新事件
    events = events[start_idx:]
    if not events:
        print("No need to update.")  # 无新事件，无需更新
        return

    # 4. 爬取新事件的详情页，补充关键字段
    def fetch_for_detail(ev):
        """
        爬取单个事件的详情页，补充特定字段
        :param ev: 单个事件字典
        """
        if ev["event_type"] == "Possible Hijack":
            # 疑似劫持事件：补充预期前缀和检测到的前缀
            detail_page = get_page(urljoin(url_index, ev["moredetail"]))
            # 匹配预期前缀（格式如"Expected prefix: 1.1.1.0/24"）
            pattern = r'Expected prefix: (.+?/\d{1,2})'
            expected = re.search(pattern, detail_page)
            if expected is not None:
                ev["expected_prefix"] = expected[1]
            else:
                print(f"unknown expected_prefix: {ev}")

            # 匹配检测到的前缀（格式如"Detected advertisement: 1.1.1.0/24"）
            pattern = r'Detected advertisement: (.+?/\d{1,2})'
            detected = re.search(pattern, detail_page)
            if detected is not None:
                ev["detected_prefix"] = detected[1]
            else:
                print(f"unknown detected_prefix: {ev}")

        elif ev["event_type"] == "BGP Leak":
            # BGP泄露事件：补充泄露前缀和泄露目标ASN
            detail_page = get_page(urljoin(url_index, ev["moredetail"]))
            # 匹配泄露前缀（格式如"Leaked prefix: 1.1.1.0/24"）
            pattern = r'Leaked prefix: (.+?/\d{1,2})'
            leaked = re.search(pattern, detail_page)
            if leaked is not None:
                ev["leaked_prefix"] = leaked[1]
            else:
                print(f"unknown leaked_prefix: {ev}")

            # 匹配泄露目标（格式如"Leaked To:<br> <li>1234"）
            pattern = r'Leaked To:\<br\>\s+\<li\>(\d+)'
            leakedto = re.search(pattern, detail_page)
            if leakedto is not None:
                ev["leaked_to"] = leakedto[1]
            else:
                print(f"unknown leaked_to: {ev}")

    # 多线程爬取详情页（最大128个线程，提升效率）
    with ThreadPoolExecutor(max_workers=128) as executor:
        executor.map(fetch_for_detail, events)

    # 5. 将新事件写入缓存文件（文件名：最新事件ID.jsonl）
    f = open(CACHE_DIR/f"{ids[-1]}.jsonl", "w")
    # 每个事件转为JSON字符串，按行存储
    f.write("\n".join([json.dumps(ev) for ev in events])+"\n")
    f.close()

    # 打印更新信息
    print(f"Update {len(events)} items")
    print(f"Latest event_id: {ids[-1]}")

# 执行缓存更新逻辑
update_cache()
