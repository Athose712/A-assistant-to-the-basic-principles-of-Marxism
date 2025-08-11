import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from mayuan_agent import MayuanQuestionAgent
from mayuan_kg_agent import MayuanKnowledgeGraphAgent
from mayuan_qa_agent import MayuanAnswerAgent  # NEW: Q&A agent
import uuid
import tempfile
import base64
from io import BytesIO
from PIL import Image
from role_agent import SocratesAgent
from dotenv import load_dotenv

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

# 配置文件上传
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', tempfile.gettempdir())
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_image(image_data):
    """保存base64编码的图片数据并返回文件路径（保留或推断正确后缀）。"""
    try:
        # 解析base64数据，并尽量获取 MIME 类型
        mime_type = None
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            header, data = image_data.split(',', 1)
            try:
                mime_type = header.split(';')[0].split(':', 1)[1]  # e.g. image/png
            except Exception:
                mime_type = None
            image_binary = base64.b64decode(data)
        else:
            image_binary = base64.b64decode(image_data)
        
        # 验证图像大小（不超过16MB，与前端/Flask设置一致）
        if len(image_binary) > 16 * 1024 * 1024:
            print("图片文件过大：超过16MB限制")
            return None
        
        # 使用Pillow打开并验证图像，顺便获取真实格式
        try:
            img = Image.open(BytesIO(image_binary))
            img.verify()  # 验证图像完整性
            real_format = img.format  # e.g. 'PNG', 'JPEG', 'WEBP'
            # 如果图片分辨率超过 4096px，则拒绝处理（与提示一致）
            if img.width > 4096 or img.height > 4096:
                print("图片分辨率过高：超过 4K 限制，拒绝处理")
                return None
        except Exception as e:
            print(f"无效的图像文件: {e}")
            return None

        # 依据 MIME/真实格式确定文件后缀
        def guess_ext(mt: str | None, fmt: str | None) -> str:
            mapping = {
                'JPEG': 'jpg', 'JPG': 'jpg', 'PNG': 'png', 'WEBP': 'webp', 'BMP': 'bmp', 'GIF': 'gif'
            }
            if fmt and fmt.upper() in mapping:
                return mapping[fmt.upper()]
            if mt and '/' in mt:
                mt_ext = mt.split('/')[-1].lower()
                if mt_ext == 'jpeg':
                    return 'jpg'
                return mt_ext
            return 'jpg'

        ext = guess_ext(mime_type, locals().get('real_format'))

        # 创建临时文件（带正确后缀），写入原始二进制
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
        temp_file.write(image_binary)
        temp_file.close()
        return temp_file.name

    except Exception as e:
        print(f"保存图片失败: {e}")
        return None

def cleanup_temp_file(file_path):
    """清理临时文件"""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"清理临时文件失败: {e}")

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

# ----- Q&A Agent -----
try:
    qa_agent = MayuanAnswerAgent()
    print("MayuanAnswerAgent loaded successfully.")
except Exception as e:
    print(f"Error loading MayuanAnswerAgent: {e}")
    qa_agent = None

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
    user_message = (data.get("message") or "").strip()
    image_data = data.get("image")  # 获取base64编码的图片数据
    response_mode = (data.get("response_mode") or "balanced").lower()
    
    # 允许只上传图片无文本
    if not user_message and not image_data:
        return jsonify({"error": "请输入文本或上传图片"}), 400

    # 处理图片数据
    image_path = None
    if image_data:
        image_path = save_uploaded_image(image_data)
        if not image_path:
            return jsonify({"error": "图片处理失败"}), 400

    response_text = ""
    try:
        # 若纯图片无文本，提供默认提示词
        if not user_message:
            user_message = "请结合图片进行分析并回答问题。"
        # 根据响应模式设置参数
        def apply_mode(agent):
            try:
                if not hasattr(agent, 'set_generation_params'):
                    return
                if response_mode == 'fast':
                    agent.set_generation_params(max_tokens=400, timeout=15, retrieval_k=3)
                elif response_mode == 'detailed':
                    agent.set_generation_params(max_tokens=1600, timeout=45, retrieval_k=7)
                else:  # balanced
                    agent.set_generation_params(max_tokens=1000, timeout=30, retrieval_k=5)
            except Exception:
                pass
        # Simple routing logic
        if any(k in user_message for k in ["知识图谱", "思维导图", "mindmap", "图谱"]):
            if kg_agent:
                print("Routing to Knowledge Graph Agent.")
                apply_mode(kg_agent)
                # 知识图谱Agent暂时不支持图片，如果有图片就提示用户
                if image_path:
                    response_text = "知识图谱生成功能暂时不支持图片输入，请使用纯文本描述您需要的知识图谱主题。"
                else:
                    response_text = kg_agent.process_request(user_message)
            else:
                response_text = "知识图谱助手未成功加载，无法处理您的请求。"
        else:
            # 先判断是否为“解答/解析”需求
            answer_keywords = [
                "解答", "答案", "解析", "请回答", "帮我回答", "帮我解答"
            ]

            # 同时检测消息是否包含 A./B./C. 等选项格式（粗略判断）
            constains_mcq_options = False
            try:
                import re
                if re.search(r"[A-DＡ-Ｄ][\.．、]\s?", user_message):
                    constains_mcq_options = True
            except Exception:
                constains_mcq_options = False

            if any(kw in user_message for kw in answer_keywords) or constains_mcq_options:
                # ---------- Q&A Agent ----------
                if qa_agent:
                    print("Routing to Q&A Agent (answer mode).")
                    apply_mode(qa_agent)
                    if image_path and hasattr(qa_agent, 'process_multimodal_request'):
                        response_text = qa_agent.process_multimodal_request(user_message, image_path)
                    else:
                        response_text = qa_agent.process_request(user_message)
                else:
                    response_text = "问答助手未成功加载，无法处理您的请求。"

            else:
                # 检测是否为出题需求
                exam_keywords = [
                    "出题", "生成题目", "选择题", "判断题", "简答题", "试题", "练习"
                ]

                if any(kw in user_message for kw in exam_keywords):
                    # ---------- Question Generation Agent ----------
                    if question_agent:
                        print("Routing to Question Generation Agent.")
                        apply_mode(question_agent)
                        if hasattr(question_agent, 'process_multimodal_request'):
                            response_text = question_agent.process_multimodal_request(user_message, image_path)
                        else:
                            if image_path:
                                response_text = "当前版本暂时不支持图片分析，请使用纯文本提问。"
                            else:
                                response_text = question_agent.process_request(user_message)
                    else:
                        response_text = "出题助手未成功加载，无法处理您的请求。"
                else:
                    # 默认走 Q&A
                    if qa_agent:
                        print("Routing to Q&A Agent (default).")
                        apply_mode(qa_agent)
                        if image_path and hasattr(qa_agent, 'process_multimodal_request'):
                            response_text = qa_agent.process_multimodal_request(user_message, image_path)
                        else:
                            response_text = qa_agent.process_request(user_message)
                    else:
                        response_text = "问答助手未成功加载，无法处理您的请求。"

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        response_text = f"处理您的请求时发生内部错误: {e}"
    finally:
        # 清理临时图片文件
        if image_path:
            cleanup_temp_file(image_path)

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
    user_message = (data.get("message") or "").strip()
    image_data = data.get("image")  # 获取base64编码的图片数据
    response_mode = (data.get("response_mode") or "balanced").lower()
    
    # 允许纯图片启动对话
    if not user_message and not image_data:
        return jsonify({"error": "请输入您想探讨的话题或上传图片"}), 400
    
    # 处理图片数据
    image_path = None
    if image_data:
        image_path = save_uploaded_image(image_data)
        if not image_path:
            return jsonify({"error": "图片处理失败"}), 400
    
    try:
        session_id = str(uuid.uuid4())
        # 应用响应模式到苏格拉底代理
        try:
            if hasattr(socrates_agent, 'set_generation_params'):
                if response_mode == 'fast':
                    socrates_agent.set_generation_params(max_tokens=400, timeout=15, retrieval_k=3)
                elif response_mode == 'detailed':
                    socrates_agent.set_generation_params(max_tokens=1600, timeout=45, retrieval_k=7)
                else:
                    socrates_agent.set_generation_params(max_tokens=1000, timeout=30, retrieval_k=5)
        except Exception:
            pass
        
        # 如果用户未提供文本但有图片，给默认提示词
        if not user_message:
            user_message = "请结合这张图片开始对话并提出苏格拉底式问题。"
        # 如果有图片，使用多模态对话功能
        if image_path and hasattr(socrates_agent, 'process_multimodal_dialogue'):
            response_data = socrates_agent.process_multimodal_dialogue(user_message, None, image_path)
        else:
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
    finally:
        # 清理临时图片文件
        if image_path:
            cleanup_temp_file(image_path)

@app.route('/continue_dialogue', methods=['POST'])
def continue_dialogue():
    if not socrates_agent:
        return jsonify({"error": "AI助手未正确初始化"}), 500
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    user_message = (data.get("message") or "").strip()
    image_data = data.get("image")  # 获取base64编码的图片数据
    response_mode = (data.get("response_mode") or "balanced").lower()
    
    if not session_id or session_id not in dialogue_sessions:
        return jsonify({"error": "会话已过期，请重新开始对话"}), 400
    if not user_message and not image_data:
        return jsonify({"error": "请输入您的回应或上传图片"}), 400
    
    # 处理图片数据
    image_path = None
    if image_data:
        image_path = save_uploaded_image(image_data)
        if not image_path:
            return jsonify({"error": "图片处理失败"}), 400
    
    try:
        current_state = dialogue_sessions[session_id]
        # 应用响应模式
        try:
            if hasattr(socrates_agent, 'set_generation_params'):
                if response_mode == 'fast':
                    socrates_agent.set_generation_params(max_tokens=400, timeout=15, retrieval_k=3)
                elif response_mode == 'detailed':
                    socrates_agent.set_generation_params(max_tokens=1600, timeout=45, retrieval_k=7)
                else:
                    socrates_agent.set_generation_params(max_tokens=1000, timeout=30, retrieval_k=5)
        except Exception:
            pass
        
        # 如果用户未提供文本但有图片，给默认提示词
        if not user_message:
            user_message = "请结合这张图片继续对话并提出苏格拉底式问题。"
        # 如果有图片，使用多模态对话功能
        if image_path and hasattr(socrates_agent, 'process_multimodal_dialogue'):
            response_data = socrates_agent.process_multimodal_dialogue(user_message, current_state, image_path)
        else:
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
    finally:
        # 清理临时图片文件
        if image_path:
            cleanup_temp_file(image_path)

@app.route('/end_dialogue', methods=['POST'])
def end_dialogue():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if session_id and session_id in dialogue_sessions:
        dialogue_sessions.pop(session_id, None)
        return jsonify({"message": "对话已结束"})
    return jsonify({"message": "会话未找到或已结束"})

def run_app():
    load_dotenv()  # 加载环境变量
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