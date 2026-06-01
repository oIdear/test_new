from flask import Flask, request, jsonify
import requests
from pathlib import Path
import json
from embedding_store import EmbeddingStore
from historical_lookup import search_historical_cases

app = Flask(__name__)

# 添加CORS支持
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# ===== RAG向量库初始化 =====
store = EmbeddingStore()

BASE_DIR = Path(__file__).resolve().parent

INDEX_PATH = BASE_DIR / "faiss_index.bin"
DATA_PATH = BASE_DIR / "index_data.json"

store.load_index(str(INDEX_PATH), str(DATA_PATH))
print("✅ RAG向量索引加载完成")

# 配置
DEEPSEEK_API_KEY = "sk-.................."
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

#Flask 端读取 jsonl
def load_alarm_by_id(group_id):
    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "post_processor" / "summary_output" / "alarms_wide_202408.jsonl"
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj["group_id"] == int(group_id):
                return obj

# 查询翻译函数：将中文查询翻译为英文
def translate_query_to_en(query_text):
    """
    使用 DeepSeek API 将中文查询翻译为英文
    用于跨语言检索场景
    """
    # 简单的语言检测：如果包含中文字符，则认为是中文
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in query_text)
    
    if not has_chinese:
        # 如果不是中文，直接返回原文
        return query_text
    
    print(f"检测到中文查询，正在翻译为英文...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    messages = [
        {
            "role": "system",
            "content": "你是一个专业的技术翻译助手。请将用户的中文技术问题准确翻译为英文，保持专业术语的准确性。只返回翻译结果，不要添加任何解释。"
        },
        {
            "role": "user",
            "content": f"请将以下中文技术问题翻译为英文：\n\n{query_text}"
        }
    ]
    
    payload = {
        "model": "deepseek-chat",  # 使用更快的 chat 模型进行翻译
        "messages": messages,
        "temperature": 0.1,  # 低温度以确保翻译一致性
        "max_tokens": 200
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        translated = response.json()["choices"][0]["message"]["content"].strip()
        print(f"翻译结果: {translated}")
        return translated
    except Exception as e:
        print(f"翻译失败: {e}，使用原始查询")
        return query_text  # 失败时回退到原文

# 调用DeepSeek-reasoner API函数
def call_deepseek_api(prompt, context):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    messages = [
        {
            "role": "system",
            "content": "你是一个BGP异常溯源分析专家，基于提供的上下文信息，对BGP异常进行分析和解释。"
        },
        {
            "role": "user",
            "content": f"上下文信息:\n{context}\n\n问题:\n{prompt}"
        }
    ]
    
    payload = {
        "model": "deepseek-reasoner",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4090
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"调用DeepSeek API时出错: {e}")
        return "API调用失败，请稍后重试。"

# pattern 代码 → 语义描述映射，用于生成纯语义查询词（去除数值噪声）
_PATTERN_DESCRIPTIONS = {
    "a1": "origin AS validation RPKI ROA route origin authorization IRR prefix legitimacy",
    "a2": "valley-free violation route leak provider customer peer AS relationship BGP policy",
    "a3": "reserved ASN unknown autonomous system bogon path anomaly invalid AS number",
    "a4": "same organization origin AS change internal routing sibling AS",
    "b1": "origin connectivity RPKI validation upstream provider customer link",
    "b2": "AS path prepending traffic engineering load balancing",
    "b3": "different upstream provider path change origin upstream diversity",
}


def _build_rag_query(alarm):
    """
    从告警中提取语义信号构建查询词。
    策略：用 pattern 语义描述替代原始数值字段，保留 RPKI 状态等有语义的字符串。
    查询词控制在 500 字符内，简洁比冗长更有效。
    """
    parts = ["BGP routing anomaly"]

    # 收集出现过的 pattern 类型（去重）
    seen_patterns = set()
    for event in alarm.get("events", []):
        for rc in event.get("route_changes", []):
            for pat_key in rc.get("patterns", {}).keys():
                seen_patterns.add(pat_key)

    # 将 pattern 代码转为语义描述
    for pat_key in sorted(seen_patterns):
        if pat_key in _PATTERN_DESCRIPTIONS:
            parts.append(_PATTERN_DESCRIPTIONS[pat_key])

    # RPKI/IRR 状态是有语义的字符串，保留
    for event in alarm.get("events", []):
        for rc in event.get("route_changes", []):
            a1 = rc.get("patterns", {}).get("a1", {})
            for field in ["origin_rpki_1", "origin_rpki_2", "origin_irr_1", "origin_irr_2"]:
                val = a1.get(field, "")
                if val:
                    parts.append(f"RPKI status {val}")
                    break  # 每条 rc 只追加一次，避免重复

    # BEAM 极端异常单独标注语义
    for event in alarm.get("events", []):
        for rc in event.get("route_changes", []):
            if rc.get("diff") == float("inf"):
                parts.append("extreme path anomaly unseen AS relationship")
                break

    query = " ".join(parts)[:500]
    print(f"RAG查询词({len(query)}字符): {query[:120]}...")
    return query


def rag_search_context(alarm, k=3):
    query = _build_rag_query(alarm)
    # hybrid=True 启用 BM25+向量+RRF+MMR 全流程
    results = store.search(query, k=k, hybrid=True)

    context_blocks = []
    for item, dist in results:
        context_blocks.append(
            f"[来源:{item.get('file','unknown')} | 相关度:{dist:.4f}]\n"
            f"{item['content'][:800]}"
        )

    # 历史异常案例精确匹配（按 ASN / 前缀查找私有案例库）
    hist_context = search_historical_cases(alarm, max_per_type=2)
    if hist_context:
        context_blocks.append("【历史异常案例匹配】\n" + hist_context)

    return "\n\n".join(context_blocks)
# 分析异常接口
@app.route("/api/analyze", methods=["POST"])
def analyze_anomaly():

    data = request.json
    gid = data["group_id"]

    alarm = load_alarm_by_id(gid)

    if not alarm:
        return {"success": False, "error": "alarm not found"}

    # ===== RAG 检索 =====
    rag_context = rag_search_context(alarm, k=3)

    print("======RAG检索结果======")
    print(rag_context[:2000])

    # ===== 构造提示词 =====
    prompt = f"""
    你是一名BGP异常检测与溯源分析专家，同时具备RFC标准、AS关系模型和路由策略分析经验。

    现在提供两类信息：
    一类是【异常观测数据】（来自BGP检测系统的结构化事件）
    一类是【检索增强上下文】（来自RFC文档、研究论文和运维知识库的相关资料）

    请基于检索证据进行约束推理（Evidence-grounded reasoning），禁止脱离证据臆测结论。
    每引用一条检索资料作为判据，必须在句末附注来源，格式严格为：[来源:文件名 | 相似度:X.XXX]
    分析目标是：对异常进行溯源解释与归因判断。

    ====================
    【异常观测数据】
    {json.dumps(alarm, indent=2)}
    ====================

    请输出一份结构化溯源分析报告，必须包含以下部分：

    【1】异常类型判定  
    - 判定属于：路由泄露 / 路由劫持 / 错误宣告 / 策略异常 / 可疑但不确定  
    - 给出判定依据

    【2】异常传播与路径特征分析  
    - AS路径变化模式
    - 是否出现 valley-free 破坏 / 非法上游传播 / 路径突变
    - 是否存在MOAS或origin变化

    【3】关键证据链（必须引用）  
    - 明确列出使用了哪些字段与RAG资料
    - 用“证据 → 推论”形式说明
    - 尽量引用RFC或研究资料中的判据依据

    【4】可能触发机制  
    - 策略误配置 / 泄露传播 / 攻击行为 / 自动聚合错误
    - 对应机制说明

    【5】影响范围评估  
    - 影响前缀范围
    - 潜在影响AS类型
    - 是否可能跨域传播

    【6】验证建议（运维可执行）  
    - 建议检查：
    - RPKI
    - IRR
    - ROA
    - AS关系
    - 上游策略
    - 历史更新流

    【7】缓解与处置建议  
    - 过滤策略
    - 前缀限制
    - 上游协调
    - RPKI验证

    【8】置信度评分  
    - 给出 0–1 之间置信度
    - 说明不确定来源

    要求：
    - 必须基于检索到的资料进行推理
    - 若证据不足要明确说明
    - 优先引用RFC或研究资料作为依据
    - 输出条理清晰
    """

    analysis = call_deepseek_api(prompt, rag_context)

    return {
        "success": True,
        "analysis": analysis
    }

# 知识问答接口
@app.route("/api/qa", methods=["POST"])
def knowledge_qa():
    data = request.json
    question = data.get("question", "")
    
    # ===== RAG 检索增强（跨语言检索）=====
    print(f"收到中文知识问答请求: {question[:100]}...")
    
    # 步骤 1: 将中文查询翻译为英文（用于检索英文向量库）
    translated_query = translate_query_to_en(question)
    
    # 步骤 2: 使用翻译后的英文查询进行混合检索
    print(f"使用英文查询词检索: {translated_query[:150]}")
    rag_results = store.search(translated_query, k=3, hybrid=True)
    
    # 步骤 3: 构建检索上下文
    context_blocks = []
    for item, dist in rag_results:
        context_blocks.append(
            f"[来源:{item.get('file','unknown')} | 相似度:{dist:.3f}]\n"
            f"{item['content'][:800]}"
        )
    
    rag_context = "\n\n".join(context_blocks) if context_blocks else "未找到相关参考资料"
    
    print(f"RAG检索到 {len(rag_results)} 个相关文档")
    if rag_context != "未找到相关参考资料":
        print(f"上下文预览: {rag_context[:2000]}...")
    
    # 步骤 4: 构造中文提示词（强制要求中文回答）
    prompt = f"""
你是一个BGP路由安全与网络运维领域的专家助手。**请严格使用中文回答用户的问题**。

请基于以下检索到的专业知识来回答用户的问题。如果检索内容与问题相关，请优先引用这些资料；如果检索内容不相关或不足以回答问题，你可以基于自己的专业知识作答，但需要明确说明。

【检索到的相关知识】
{rag_context}

【用户问题】
{question}

请提供专业、准确、条理清晰的中文回答。如果涉及技术概念，请适当解释；如果涉及操作建议，请给出具体步骤。
"""
    
    # 调用DeepSeek API
    answer = call_deepseek_api(prompt, rag_context)
    
    return jsonify({
        "success": True,
        "answer": answer,
        "retrieved_docs_count": len(rag_results),
        "has_rag_context": rag_context != "未找到相关参考资料",
        "translated_query": translated_query  # 返回翻译后的英文查询，便于调试
    })

# 健康检查接口
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "RAG系统运行正常"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
