# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在此代码仓库中工作所需的指导信息。

## 项目概述

语义感知路由异常检测系统——USENIX Security 2024 论文《Learning with Semantics: Towards a Semantics-Aware Routing Anomaly Detection System》的研究代码库。系统接入真实 BGP 路由数据，训练 AS 嵌入模型（BEAM），检测路由变更并对其异常程度打分，最终生成 HTML 异常报告。另有一套基于 RAG 的 Flask API 层，通过 DeepSeek LLM 对检测到的异常进行自然语言分析。

## 环境配置

需要 Python ≥ 3.8，使用 conda 创建环境：

```bash
conda create -n beam python=3.8 numpy pandas scipy tqdm joblib click pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda activate beam
pip install sentence-transformers faiss-cpu PyPDF2 langchain flask requests
```

还需将 [BGPdump](https://github.com/RIPE-NCC/bgpdump)（用于解析 MRT 格式路由数据）的二进制文件软链接到 `data/routeviews/bgpd`。

## 五步流水线

所有脚本均支持 `--help` 查看参数。在仓库根目录下、激活 `beam` 环境后执行。

**第一步：训练 BEAM 模型**
```bash
python BEAM_engine/train.py --serial 2 --time 20240801 --Q 10 --dimension 128 --epoches 1000 --device 0 --num-workers 10
```
模型保存至 `BEAM_engine/models/<模型名>/`（link.emb、node.emb、rela.emb）。CAIDA AS 关系数据首次使用时自动下载至 `data/caida_as_rel/`。

**第二步：检测路由变更**
```bash
python routing_monitor/detect_route_change_routeviews.py --collector wide --year 2024 --month 8
```
结果输出至 `routing_monitor/detection_result/<collector>/`。RouteViews MRT 数据首次使用时自动下载至 `data/routeviews/updates/`。

**第三步：量化路径差异**
```bash
python anomaly_detector/BEAM_diff_evaluator_routeviews.py --collector wide --year 2024 --month 8 --beam-model 20240801.as-rel2.1000.10.128
```
BEAM 差异分数输出至路由变更目录下的 `BEAM_metric/`。

**第四步：检测异常**
```bash
python anomaly_detector/report_anomaly_routeviews.py --collector wide --year 2024 --month 8
```
异常告警（CSV + JSON）输出至 `reported_alarms/`。

**第五步：生成报告**
```bash
# 先对告警进行属性后处理（RPKI/IRR/WHOIS 增强）：
python post_processor/alarm_postprocess_routeviews.py --collector wide --year 2024 --month 8
# 再生成 HTML 报告和 JSONL 摘要：
python post_processor/summary_routeviews.py --collector wide --year 2024 --month 8
# 或使用封装脚本一键执行：
./generate_report.sh wide 2024 8
```
HTML 报告保存至 `post_processor/html/`，JSONL 摘要保存至 `post_processor/summary_output/`。

## RAG 系统（rag_system/）

基于 FAISS 向量索引的 Flask API，支持对 BGP 知识文档（RFC、研究论文、`bgp-embedding-data/` 中的 CSV 异常数据）进行 LLM 增强问答与异常分析。

**构建向量索引**（文档变更时重新执行）：
```bash
cd rag_system
python data_parser.py       # 生成 parsed_data.json
python embedding_store.py   # 生成 faiss_index.bin 和 index_data.json
```

**启动 API 服务**（端口 5000）：
```bash
cd rag_system && python app.py
```

**API 接口：**
- `POST /api/analyze` — RAG 增强异常分析，请求体：`{"group_id": <int>}`
- `POST /api/qa` — 跨语言知识问答（中文查询 → 翻译为英文 → 向量检索 → 中文回答），请求体：`{"question": "<问题>"}`
- `GET /api/health` — 健康检查

`post_processor/html/` 中的 HTML 报告会直接请求 `http://127.0.0.1:5000/api/analyze`，因此使用报告界面时必须保持 Flask 服务运行。

**⚠️ 安全提示：** `DEEPSEEK_API_KEY` 当前硬编码在 [rag_system/app.py:29](rag_system/app.py#L29)，提交或共享代码前应迁移至 `.env` 文件。

## 架构说明

- **数据流向**：CAIDA AS 关系数据 → BEAM 模型训练 → [RouteViews MRT 数据 → 路由变更检测] → BEAM 差异评分 → 异常告警生成 → 后处理（RPKI/IRR/WHOIS 增强）→ HTML 报告 → RAG+LLM 分析
- **BEAM 模型**（[BEAM_engine/BEAM_model.py](BEAM_engine/BEAM_model.py)）：对 AS 关系图进行图嵌入；嵌入向量之间的距离量化了新 AS 路径相对于训练关系的"异常程度"。
- **路由监视器**（[routing_monitor/monitor.py](routing_monitor/monitor.py)）：用 Trie 结构维护全局路由表，按月顺序处理 MRT 更新。
- **异常检测器**（[anomaly_detector/utils.py](anomaly_detector/utils.py)）：加载 BEAM 嵌入并计算每条路由变更的差异分数；`report_anomaly_routeviews.py` 将结果聚合为告警组。
- **后处理器**：通过 `rpki_validator.py`、`irr_validator.py`、`whois_lookup.py` 对告警进行属性增强，各模块均有本地文件缓存。
- **RAG 流水线**：`data_parser.py` 将 PDF/CSV 切分为文本块 → `embedding_store.py` 使用 `all-MiniLM-L6-v2` 构建 L2 归一化的 FAISS `IndexFlatIP` 索引（余弦相似度）→ `app.py` 从告警 JSON 字段构造多维查询词并检索 top-k 文本块作为 LLM 上下文。

## 关键数据路径

| 用途 | 路径 |
|---|---|
| CAIDA AS 关系数据 | `data/caida_as_rel/serial-{1,2}/` |
| RouteViews MRT 更新数据 | `data/routeviews/updates/<collector>/` |
| BGPdump 二进制 | `data/routeviews/bgpd` |
| BEAM 训练模型 | `BEAM_engine/models/` |
| 路由变更检测结果 | `routing_monitor/detection_result/<collector>/` |
| BEAM 差异分数 | `routing_monitor/detection_result/<collector>/BEAM_metric/` |
| 异常告警 | `routing_monitor/detection_result/<collector>/reported_alarms/` |
| 增强后的告警 | `routing_monitor/detection_result/<collector>/reported_alarms.flags/` |
| HTML 报告 | `post_processor/html/` |
| 告警 JSONL（供 RAG API 使用） | `post_processor/summary_output/alarms_<collector>_<YYYYMM>.jsonl` |
| RAG 知识文档 | `bgp-embedding-data/` |
| FAISS 向量索引 | `rag_system/faiss_index.bin` + `rag_system/index_data.json` |
