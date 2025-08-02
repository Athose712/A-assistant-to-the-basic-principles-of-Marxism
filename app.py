import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
from mayuan_agent import MayuanQuestionAgent
from mayuan_kg_agent import MayuanKnowledgeGraphAgent
import uuid
from role import SocratesAgent

# ---------- Knowledge-Graph Agent 包装 ----------
# 为了与 QuestionAgent 保持统一的调用接口，
# 这里包装一个 process_request 方法，对用户输入做简单的主题抽取后
# 调用 MayuanKnowledgeGraphAgent.build_knowledge_graph 生成 Mermaid 图。


class MayuanKGAgent(MayuanKnowledgeGraphAgent):
    """Knowledge-Graph Agent 的薄包装，提供统一的 process_request 接口。"""

    def _extract_topic(self, user_input: str) -> str:
        """从用户输入中提取知识图谱主题。

        策略：
        1. 去除常见触发关键词（知识图谱/思维导图/mindmap 等）。
        2. 删除标点与前后空白。
        3. 若为空则回退使用完整输入。
        """

        trigger_keywords = [
            "知识图谱",
            "思维导图",
            "mindmap",
            "图谱",
            "生成",
            "制作",
            "构建",
            "画",
            "帮我",
            "请",
            "关于",
            "：",
            ":",
        ]

        topic = user_input
        for kw in trigger_keywords:
            topic = topic.replace(kw, "")

        # 移除多余空格和常见中文标点
        topic = topic.strip().lstrip("，,。 、")

        return topic if topic else user_input

    def process_request(self, user_input: str) -> str:
        topic = self._extract_topic(user_input)
        return self.build_knowledge_graph(topic)


app = Flask(__name__)

# Load Agents
# It's better to load these once at startup.
try:
    question_agent = MayuanQuestionAgent()
    print("MayuanQuestionAgent loaded successfully.")
except Exception as e:
    print(f"Error loading MayuanQuestionAgent: {e}")
    question_agent = None

try:
    kg_agent = MayuanKGAgent()
    print("MayuanKGAgent loaded successfully.")
except Exception as e:
    print(f"Error loading MayuanKGAgent: {e}")
    kg_agent = None

# ----- Role Play Agent -----
dialogue_sessions = {}

try:
    socrates_agent = SocratesAgent()
    print("SocratesAgent initialized.")
except Exception as e:
    print(f"Error initializing SocratesAgent: {e}")
    socrates_agent = None


@app.route('/chat_ui')
def chat_ui():
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat', methods=['POST'])
def chat():
    # 无论 client Content-Type 如何，安全地解析 JSON，避免 request.json 为 None
    data = request.get_json(silent=True) or {}
    user_message = data.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response_text = ""
    try:
        # Simple routing logic
        if any(k in user_message for k in ["知识图谱", "思维导图", "mindmap", "图谱"]):
            if kg_agent:
                print("Routing to Knowledge Graph Agent.")
                response_text = kg_agent.process_request(user_message)
            else:
                response_text = "知识图谱助手未成功加载，无法处理您的请求。"
        else:
            if question_agent:
                print("Routing to Question Generation Agent.")
                response_text = question_agent.process_request(user_message)
            else:
                response_text = "出题助手未成功加载，无法处理您的请求。"

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        response_text = f"处理您的请求时发生内部错误: {e}"

    return jsonify({"response": response_text})

# ---------------- 角色扮演端点 ----------------

@app.route('/role')
def role_chat_page():
    """角色扮演页面"""
    return render_template('role_chat.html')

@app.route('/start_dialogue', methods=['POST'])
def start_dialogue():
    if not socrates_agent:
        return jsonify({"error": "AI助手未正确初始化"}), 500
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "请输入您想探讨的话题"}), 400
    try:
        session_id = str(uuid.uuid4())
        response_data = socrates_agent.process_dialogue(user_message, None)
        if response_data["status"] == "error":
            return jsonify({"error": response_data["response"]}), 500
        dialogue_sessions[session_id] = response_data["state"]
        return jsonify({
            "session_id": session_id,
            "response": response_data["response"],
            "character": response_data["state"]["simulated_character"],
            "topic": response_data["state"]["current_topic"],
            "turn_count": response_data["state"]["turn_count"]
        })
    except Exception as e:
        print(f"Error starting dialogue: {e}")
        return jsonify({"error": "内部错误"}), 500

@app.route('/continue_dialogue', methods=['POST'])
def continue_dialogue():
    if not socrates_agent:
        return jsonify({"error": "AI助手未正确初始化"}), 500
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    user_message = data.get("message", "").strip()
    if not session_id or session_id not in dialogue_sessions:
        return jsonify({"error": "会话已过期，请重新开始对话"}), 400
    if not user_message:
        return jsonify({"error": "请输入您的回应"}), 400
    try:
        current_state = dialogue_sessions[session_id]
        response_data = socrates_agent.process_dialogue(user_message, current_state)
        if response_data["status"] == "error":
            return jsonify({"error": response_data["response"]}), 500
        dialogue_sessions[session_id] = response_data["state"]
        return jsonify({
            "response": response_data["response"],
            "character": response_data["state"]["simulated_character"],
            "topic": response_data["state"]["current_topic"],
            "turn_count": response_data["state"]["turn_count"]
        })
    except Exception as e:
        print(f"Error continuing dialogue: {e}")
        return jsonify({"error": "内部错误"}), 500

@app.route('/end_dialogue', methods=['POST'])
def end_dialogue():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if session_id and session_id in dialogue_sessions:
        dialogue_sessions.pop(session_id, None)
        return jsonify({"message": "对话已结束"})
    return jsonify({"message": "会话未找到或已结束"})

def run_app():
    # Before running, make sure the API key is set if your agents need it.
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("\n" + "="*50)
        print("❌ 警告: 未找到 DASHSCOPE_API_KEY 环境变量。")
        print("如果您的 Agent 需要调用 DashScope API，请务必进行设置。")
        print("您可以临时在终端运行:")
        print("  $env:DASHSCOPE_API_KEY='your_api_key_here'  (Windows PowerShell)")
        print("  export DASHSCOPE_API_KEY='your_api_key_here' (Linux/Mac)")
        print("="*50 + "\n")
        
    app.run(debug=True, port=5001)

if __name__ == '__main__':
    run_app() 