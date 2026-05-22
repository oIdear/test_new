# RAG 系统升级记录

**日期：** 2026-05-22  
**涉及文件：** `rag_system/embedding_store.py`、`rag_system/app.py`、`rag_system/data_parser.py`、`bgp-embedding-data/`

---

## 一、知识库扩充

### 新增文档

在 `bgp-embedding-data/` 目录中补充了 3 份 RFC 纯文本文档：

| 文件 | 内容 |
|---|---|
| `rfc9234.txt` | 路由泄露防御机制（BGP Roles / OTC 属性） |
| `rfc6811.txt` | BGP 前缀起源验证（RPKI 路由起源验证 ROV） |
| `rfc8893.txt` | RPKI 路由起源授权（ROA 签发与验证） |

### 文档解析支持（`data_parser.py`）

新增 `parse_txt()` 函数，支持解析 `.txt` 格式的 RFC 纯文本文件，沿用与 PDF 相同的 LangChain `RecursiveCharacterTextSplitter` 分块策略（chunk_size=1000，overlap=150）。

### 知识库规模变化

| 指标 | 升级前 | 升级后 |
|---|---|---|
| 文档块总数 | ~630 | 3393 |
| RFC 覆盖 | rfc4271、rfc7908、rfc8205 | +rfc9234、rfc6811、rfc8893 |
| 文件类型 | PDF、CSV | PDF、CSV、TXT |

---

## 二、Embedding 模型升级

**变更：** `all-MiniLM-L6-v2` → `BAAI/bge-base-en-v1.5`

| 对比项 | 旧模型 | 新模型 |
|---|---|---|
| 向量维度 | 384 | 768 |
| MTEB 检索排名 | 中等 | 前列 |
| 检索优化 | 无 | 支持查询前缀激活检索模式 |

bge 系列模型在检索时需要对 query 添加专用前缀，已在 `_add_query_prefix()` 中处理：

```python
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
```

文档侧（build_index）编码不加前缀，仅 query 侧加，符合 bge 模型的使用规范。

---

## 三、混合检索（BM25 + 向量 + RRF + MMR）

### 依赖

```bash
pip install rank-bm25
```

### 流程说明

`embedding_store.py` 的 `search()` 方法新增完整的四阶段混合检索流程：

```
query
  ├─ 向量检索（FAISS IndexFlatIP，余弦相似度）→ top-20 候选
  ├─ BM25 检索（BM25Okapi，关键词匹配）      → top-20 候选
  ↓
  RRF 融合（倒数排名融合，c=60）             → top-k*3 合并结果
  ↓
  MMR 去重（λ=0.6，Jaccard 词重叠度）       → 最终 top-k 结果
```

**RRF（Reciprocal Rank Fusion）**：将两路检索的排名转为分数（`1/(c+rank)`）后相加，c=60 是业界常用平滑参数，避免头部排名权重过高。

**MMR（Maximal Marginal Relevance）**：在保证相关性的同时最大化结果多样性，避免返回 3 个来自同一文档的相似片段。λ=0.6 偏向相关性，0.4 权重用于惩罚与已选结果重叠度高的候选。

### 接口向下兼容

```python
# 启用混合检索（默认）
store.search(query, k=3, hybrid=True)

# 退化为纯向量检索（兼容旧调用）
store.search(query, k=3, hybrid=False)
```

---

## 四、查询词重构

### 问题

旧方案将 AS 号、时间戳、IP 前缀、BEAM 差异分数等数值字段全部拼入查询词（最长 2000 字符）。这些数值在向量空间中没有语义贡献，反而稀释了核心语义信号。

### 方案

在 `app.py` 中引入 `_PATTERN_DESCRIPTIONS` 字典，将 pattern 代码映射为语义描述：

```python
_PATTERN_DESCRIPTIONS = {
    "a1": "origin AS validation RPKI ROA route origin authorization IRR prefix legitimacy",
    "a2": "valley-free violation route leak provider customer peer AS relationship BGP policy",
    "a3": "reserved ASN unknown autonomous system bogon path anomaly invalid AS number",
    "a4": "same organization origin AS change internal routing sibling AS",
    "b1": "origin connectivity RPKI validation upstream provider customer link",
    "b2": "AS path prepending traffic engineering load balancing",
    "b3": "different upstream provider path change origin upstream diversity",
}
```

查询词长度从最长 2000 字符压缩至 **500 字符**，信噪比显著提升。

保留的有效语义字段：RPKI 状态字符串（如 `valid`、`invalid`、`not-found`）、极端异常标注。

---

## 五、重建索引

完成以上改动后，依次执行以下命令重建索引：

```bash
cd rag_system

# 步骤1：重新解析所有文档（含新增 RFC txt）
python data_parser.py

# 步骤2：用新模型重建向量索引 + BM25 索引
python embedding_store.py
```

构建结果：

```
文档块总数：3393
向量维度：768
向量索引：faiss_index.bin
文档数据：index_data.json
```

> 注意：旧的 `faiss_index.bin` 是 384 维（all-MiniLM），新索引是 768 维（bge），两者不兼容，必须完整重建。

---

## 六、改动文件汇总

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `bgp-embedding-data/rfc9234.txt` | 新增 | 路由泄露防御 RFC |
| `bgp-embedding-data/rfc6811.txt` | 新增 | RPKI ROV RFC |
| `bgp-embedding-data/rfc8893.txt` | 新增 | RPKI ROA RFC |
| `rag_system/data_parser.py` | 修改 | 新增 `parse_txt()` 及调用分支 |
| `rag_system/embedding_store.py` | 重写 | 换模型、BM25、RRF、MMR |
| `rag_system/app.py` | 修改 | 查询词去噪、pattern 语义映射、hybrid 调用 |
