from flask import Flask, request, jsonify
import requests
from pathlib import Path
import json
from embedding_store import EmbeddingStore

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
DEEPSEEK_API_KEY = "sk-18be0971348e40b6b5ff9ffcfe19a63e"
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

def rag_search_context(alarm, k=3):

    query_parts = ["BGP anomaly"]

    for ev in alarm.get("events", []):
        for rc in ev.get("route_changes", []):
            query_parts.append(rc.get("path1",""))
            query_parts.append(rc.get("path2",""))

    query_parts.append("route leak hijack misconfiguration RPKI IRR")

    query = " ".join(query_parts)[:800]  # 防止过长

    print("RAG查询词:", query[:200])

    results = store.search(query, k=k)

    context_blocks = []
    for item, dist in results:
        context_blocks.append(
            f"[来源:{item.get('file','unknown')} | 相似度:{dist:.3f}]\n"
            f"{item['content'][:800]}"
        )

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
    print(rag_context[:1000])

    #print("======发送给AI的异常JSON======")
    #print(json.dumps(alarm, indent=2))

    # ===== 构造提示词 =====
    prompt = f"""
    你是一名BGP异常检测与溯源分析专家，同时具备RFC标准、AS关系模型和路由策略分析经验。

    现在提供两类信息：
    一类是【异常观测数据】（来自BGP检测系统的结构化事件）
    一类是【检索增强上下文】（来自RFC文档、研究论文和运维知识库的相关资料）

    请基于检索证据进行约束推理（Evidence-grounded reasoning），禁止脱离证据臆测结论。
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
    
    # 调用DeepSeek API
    answer = call_deepseek_api(question, "")
    
    return jsonify({
        "success": True,
        "answer": answer
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