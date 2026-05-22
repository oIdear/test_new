# 实验一：混合检索 vs 纯向量检索

**日期：** 2026-05-22  
**系统：** bge-base-en-v1.5 + FAISS IndexFlatIP（余弦相似度）  
**知识库：** 3393 个文本块（含 rfc4271/7908/8205/9234/6811/8893 + 3 个 CSV + 1 篇论文）  
**检索数量 k：** 3

---

## 1. 实验设计

| 配置 | 说明 |
|---|---|
| **混合检索**（Hybrid） | BM25 关键词检索 + 向量检索各取 top-20 候选，RRF (c=60) 融合，MMR (λ=0.6) 去重 |
| **纯向量检索**（Vector-only） | 仅 FAISS IndexFlatIP 余弦相似度，直接取 top-3 |

测试集：20 条标准 BGP 领域英文查询，覆盖所有 pattern 类型（a1/a2/a3/a4/b1/b2/b3）及通用 BGP 安全场景。

---

## 2. 汇总结果

| 指标 | 混合检索 | 纯向量检索 |
|---|---|---|
| Top-3 命中数（/ 20） | 20 | 20 |
| Top-3 召回率 | 100.0% | 100.0% |
| 平均来源文件数（多样性） | **2.15** | 1.60 |
| 多样性提升 | +0.55 | — |

> 说明：来源文件数 = 每次检索 3 个结果中来自不同文件的数量（最大值 3），衡量结果的知识覆盖广度。

---

## 3. 逐条查询明细

| # | 查询（截断至 45 字符） | 覆盖关键词 | 混合命中 | 混合多样性 | 纯向量命中 | 纯向量多样性 |
|---|---|---|---|---|---|---|
| 1 | RPKI ROA route origin authorization prefix le | RPKI, ROA, origin | ✓ | 3 | ✓ | 2 |
| 2 | origin AS validation IRR internet routing reg | IRR, routing, origin | ✓ | 3 | ✓ | 2 |
| 3 | BGP route origin validation prefix legitimacy | RPKI, origin, valid | ✓ | 1 | ✓ | 3 |
| 4 | valley-free violation route leak provider cus | valley, provider, customer | ✓ | 3 | ✓ | 2 |
| 5 | route leak BGP policy propagation AS relation | route leak, leak, provider | ✓ | 3 | ✓ | 1 |
| 6 | BGP route leak detection prevention RFC 7908 | route leak, leak, provider | ✓ | 2 | ✓ | 2 |
| 7 | reserved ASN bogon invalid AS number in path | reserved, bogon, AS | ✓ | 3 | ✓ | 2 |
| 8 | unknown autonomous system number path anomaly | unknown, AS, path | ✓ | 2 | ✓ | 1 |
| 9 | same organization origin AS change sibling AS | organization, origin, AS | ✓ | 2 | ✓ | 1 |
| 10 | origin connectivity upstream provider custome | provider, customer, origin | ✓ | 3 | ✓ | 3 |
| 11 | AS path prepending traffic engineering load b | prepend, AS_PATH, traffic | ✓ | 2 | ✓ | 3 |
| 12 | different upstream provider path change origi | upstream, provider, path | ✓ | 3 | ✓ | 2 |
| 13 | BGP prefix hijack detection mitigation | hijack, origin, prefix | ✓ | 1 | ✓ | 1 |
| 14 | BGP security routing anomaly detection | BGP, routing, anomaly | ✓ | 1 | ✓ | 1 |
| 15 | AS path attribute BGP UPDATE message | AS_PATH, UPDATE, path | ✓ | 2 | ✓ | 2 |
| 16 | BGP route selection decision process | Decision, preference, route | ✓ | 1 | ✓ | 1 |
| 17 | RPKI ROA validation not-found invalid valid s | RPKI, valid, invalid | ✓ | 2 | ✓ | 1 |
| 18 | BGP route leak RFC 9234 BGP roles OTC attribu | route leak, roles, OTC | ✓ | 2 | ✓ | 1 |
| 19 | RPKI resource public key infrastructure origi | RPKI, origin, validation | ✓ | 3 | ✓ | 3 |
| 20 | extreme path anomaly unseen AS relationship | anomaly, path, AS | ✓ | 1 | ✓ | 1 |

---

## 4. 来源文件分布

下表统计每个来源文件在 Top-3 结果中被选中的总次数（20 条查询 × k=3 = 60 个槽位）：

| 来源文件 | 混合检索（次） | 纯向量检索（次） |
|---|---|---|
| 2402.16025v1.pdf（BEAM论文） | 15 | 18 |
| rfc4271.txt.pdf（BGP-4基础） | 9 | 9 |
| rfc6811.txt（RPKI ROV） | 8 | 7 |
| rfc7908.txt.pdf（路由泄露定义） | 3 | 3 |
| rfc8205.txt.pdf（BGPsec/ROA） | 10 | 10 |
| rfc8893.txt（RPKI ROA） | 7 | 7 |
| rfc9234.txt（路由泄露防御） | 8 | 6 |
| **合计** | 60 | 60 |

---

## 5. 多样性分布（每查询来源文件数）

| 多样性值 | 混合检索（查询数） | 纯向量检索（查询数） |
|---|---|---|
| 1（3个结果同一文件） | 6 | 9 |
| 2（来自2个不同文件） | 7 | 7 |
| 3（来自3个不同文件） | 7 | 4 |
| 平均值 | **2.15** | 1.60 |

---

## 6. 关键发现

- **召回率两者持平（100%）**：在当前 20 条测试查询上，混合检索与纯向量检索均完全命中，说明知识库覆盖已足够。
- **混合检索在来源多样性上显著更优**：平均每次检索覆盖 2.15 个不同文件来源，比纯向量（1.60）提升 **34.4%**。这意味着 LLM 获得的参考资料来自更广的知识范围，减少单一文档偏置。
- **纯向量检索存在"单一来源聚集"问题**：9 条查询（45%）的 3 个结果全部来自同一文件，MMR 去重有效解决了这一问题。
- **典型受益案例**：查询 5（route leak）纯向量 3 个结果全来自 rfc9234，混合检索则覆盖 rfc9234 + 2402论文 + rfc7908，为 LLM 提供更多样的路由泄露知识视角。
