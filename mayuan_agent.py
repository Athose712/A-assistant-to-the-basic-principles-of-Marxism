"""
马克思主义基本原理智能出题 Agent
基于 LangGraph 和 LangChain 框架构建
"""

import os
import re
import dashscope
from typing import Dict, List, TypedDict, Optional
from langchain_community.vectorstores import FAISS
from langchain_dashscope.embeddings import DashScopeEmbeddings
from langchain_dashscope import ChatDashScope
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import ChatResult, ChatGeneration
from langgraph.graph import StateGraph, END

# 重要：运行前必须设置环境变量
# 方法1：在终端中运行
#   export DASHSCOPE_API_KEY='your_api_key_here'      # Linux/Mac
#   $env:DASHSCOPE_API_KEY='your_api_key_here'        # Windows PowerShell
# 方法2：取消下面这行注释并填入您的API Key
# ——重要——
# 1. 先在环境变量中写入 API-Key，供后续子进程或其他库使用
# 2. 再显式赋值给 dashscope.api_key，避免在导入 dashscope 后才设置环境变量导致 SDK 读取不到的问题

#os.environ["DASHSCOPE_API_KEY"] = "sk-67eb31fc296f46728913a60ad6c03e32"  # 请替换为您的实际API Key并取消注释

# 让 dashscope SDK 立刻获取到正确的 key（import dashscope 已经在上面执行过）
import dashscope as _ds_internal
_ds_internal.api_key = os.environ["DASHSCOPE_API_KEY"]

#from langchain_core.chat_models import BaseChatModel
# 这是新的、正确的导入路径
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage, AIMessageChunk
from typing import Any, List, Iterator, Optional

# 自定义一个能正常工作的DashScope封装，以替代有问题的ChatDashScope
class CustomChatDashScope(BaseChatModel):
    """
    一个自定义的、能稳定工作的DashScope聊天模型封装，
    它直接使用底层的、我们已验证成功的 dashscope 官方库。
    """
    model: str = "qwen-turbo"
    temperature: float = 0.7
    # 您可以在这里添加更多希望控制的参数，如 top_p 等

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        # 将LangChain的消息格式转换为DashScope需要的格式
        # DashScope 的消息格式是 {'role': 'user'/'assistant'/'system', 'content': '...'}
        prompt_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_messages.append({'role': 'system', 'content': msg.content})
            elif isinstance(msg, HumanMessage):
                prompt_messages.append({'role': 'user', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                prompt_messages.append({'role': 'assistant', 'content': msg.content})

        # 使用我们已经验证过能正常工作的官方SDK发起调用
        response = dashscope.Generation.call(
            model=self.model,
            messages=prompt_messages,
            result_format='message',  # 确保返回的是消息格式
            temperature=self.temperature,
            stream=False,  # 明确使用非流式调用，避免返回生成器对象
            **kwargs
        )

        # DashScope 在 stream=False 时返回 GenerationResponse 对象；
        # 如果用户显式请求流式 (stream=True)，则返回一个生成器。

        # 1) 非流式：直接取 status_code / output
        if hasattr(response, "status_code"):
            if response.status_code == 200:  # type: ignore[attr-defined]
                ai_content = response.output.choices[0]["message"]["content"]  # type: ignore[attr-defined]
                return AIMessage(content=ai_content)
            # 非 200 直接抛错
            raise Exception(
                f"DashScope API Error: Code {getattr(response, 'code', 'unknown')}, Message: {getattr(response, 'message', 'unknown')}"  # type: ignore[attr-defined]
            )

        # 2) 流式：把所有 chunk 拼接为完整字符串
        if hasattr(response, "__iter__"):
            content_chunks: List[str] = []
            for chunk in response:
                try:
                    content_chunks.append(chunk.choices[0].delta.content)
                except Exception:
                    # 防御性：某些 chunk 结构可能不同
                    pass
            return AIMessage(content="".join(content_chunks))

        # 3) 未知返回类型
        raise Exception("DashScope 返回了未知的响应类型，无法解析。")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        ai_msg = self._call(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    @property
    def _llm_type(self) -> str:
        """返回一个自定义的LLM类型名称"""
        return "custom_chat_dashscope_wrapper"

class GraphState(TypedDict):
    """定义图状态，管理整个工作流中的数据状态"""
    user_input: str  # 用户输入的原始指令
    topic: str  # 解析出的出题主题
    num_questions: int  # 题目总数量
    difficulty: str  # 难度等级
    question_type: str  # 主题型：选择题/判断题/简答题/混合
    question_type_counts: Dict[str, int]  # 各题型对应的数量，用于混合题型支持
    retrieved_docs: List[str]  # 检索到的相关文档内容
    generated_questions: str  # 生成的最终题目
    error_message: Optional[str]  # 错误信息


class MayuanQuestionAgent:
    """马克思主义基本原理智能出题 Agent"""
    
    def __init__(self):
        """初始化 Agent"""
        # 检查 API Key
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise Exception("请设置 DASHSCOPE_API_KEY 环境变量")
        
        try:
            self.embeddings = DashScopeEmbeddings(model="text-embedding-v2")
            print("Embedding 模型初始化成功")
        except Exception as e:
            raise Exception(f"Embedding 模型初始化失败: {e}")
        
        try:
            self.llm = CustomChatDashScope(
                model="qwen-max",
                temperature=0.7
            )
            print("LLM 模型初始化成功")
        except Exception as e:
            raise Exception(f"LLM 模型初始化失败: {e}")
        
        self.vectorstore = None
        self.graph = None
        self._load_knowledge_base()
        self._build_graph()
    
    def _load_knowledge_base(self):
        """加载本地向量知识库"""
        try:
            print("正在加载向量知识库...")
            self.vectorstore = FAISS.load_local(
                "database_agent_mayuan", 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("向量知识库加载成功！")
        except Exception as e:
            print(f"加载向量知识库失败: {e}")
            print("Agent将在没有知识库的情况下运行（功能受限）")
            self.vectorstore = None
    
    def parse_input_node(self, state: GraphState) -> Dict:
        """解析用户输入节点 - 提取主题、数量和难度"""
        print("正在解析用户输入...")
        
        user_input = state["user_input"]
        
        # ---------- 提取题目数量 ----------
        # 1) 先尝试同时捕获“数量+题型”模式，如“2道选择题”
        type_count_pattern = r"""(\d+)\s*(?:道|题|个)[^\u4e00-\u9fa5]*?(选择题|判断题|简答题|材料\s*分析题?)"""
        # 为了处理“材料 分析题”中可能存在的空格，先去除连续空白后再匹配
        compact_input = re.sub(r"\s+", "", user_input)
        mixed_matches = re.findall(type_count_pattern, compact_input)

        question_type_counts: Dict[str, int] = {}
        # 统计各题型数量
        for num_str, q_type_raw in mixed_matches:
            # 归一化：将“材料分析（题）”统一视为“简答题”
            q_type_raw = q_type_raw.replace("材料分析题", "简答题").replace("材料分析", "简答题")
            count = int(num_str)
            question_type_counts[q_type_raw] = question_type_counts.get(q_type_raw, 0) + count

        # 2) 如果没匹配到“数量+题型”，退回到只提取数字
        if question_type_counts:
            num_questions = sum(question_type_counts.values())
        else:
            num_pattern_simple = r"(\d+)\s*(?:道|题|个)"
            numbers = re.findall(num_pattern_simple, user_input)
            num_questions = sum(int(n) for n in numbers) if numbers else 5  # 默认 5 道

        # ---------- 提取难度 ----------
        difficulty = "中等"  # 默认值
        if "简单" in user_input or "容易" in user_input or "基础" in user_input:
            difficulty = "简单"
        elif "困难" in user_input or "难" in user_input or "高级" in user_input:
            difficulty = "困难"
        elif "中等" in user_input or "中级" in user_input:
            difficulty = "中等"

        # ---------- 提取题目类型 ----------
        detected_types = []
        if "选择题" in user_input:
            detected_types.append("选择题")
        if "判断题" in user_input:
            detected_types.append("判断题")
        if re.search(r"材料\s*分析", user_input) or "简答" in user_input or "简答题" in user_input:
            detected_types.append("简答题")

        # 归一化与去重
        detected_types = list(dict.fromkeys(detected_types))

        # 如果 question_type_counts 为空，但检测到了题型列表，则为这些题型平均分配数量
        if not question_type_counts and detected_types:
            avg_count = max(1, num_questions // len(detected_types))
            for qt in detected_types:
                question_type_counts[qt] = avg_count

        # 决定主题型
        if len(question_type_counts) == 1:
            question_type = next(iter(question_type_counts))
        elif len(question_type_counts) > 1:
            question_type = "混合"
        else:
            # 默认选择题
            question_type = "选择题"
            question_type_counts = {"选择题": num_questions}

        # ---------- 提取主题（支持多主题） ----------
        detected_topics: List[str] = []

        # 1) 通过常见关键词列表直接匹配
        common_topics = [
            "唯物辩证法", "历史唯物主义", "马克思主义哲学", "认识论",
            "实践观", "矛盾论", "否定之否定", "质量互变", "联系",
            "发展", "本质与现象", "内容与形式", "原因与结果",
            "必然与偶然", "可能与现实", "社会存在", "社会意识",
            "辩证唯物主义"
        ]
        for t in common_topics:
            if t in user_input:
                detected_topics.append(t)

        # 2) 通过形如“关于XX的”短语补充主题
        about_matches = re.findall(r"关于(.*?)的", user_input)
        for raw in about_matches:
            cleaned = re.sub(r"(简单|容易|基础|中等|中级|困难|难|高级|选择题|判断题|简答题|材料\s*分析题?|题目|\s)", "", raw)
            cleaned = cleaned.strip()
            if cleaned:
                detected_topics.append(cleaned)

        # 去重，保持顺序
        seen = set()
        ordered_topics: List[str] = []
        for t in detected_topics:
            if t not in seen:
                ordered_topics.append(t)
                seen.add(t)

        # 3) 回退：若仍未检测到主题，则用简化后的整句或默认主题
        if not ordered_topics:
            fallback_topic = re.sub(r'\d+道|\d+题|\d+个|请|给我|出|关于|的|简单|中等|困难|选择题|判断题|简答题|材料\s*分析题?|题目', '', user_input)
            fallback_topic = fallback_topic.strip()
            ordered_topics = [fallback_topic if fallback_topic else "马克思主义基本原理"]

        # 使用分号连接多主题，便于后续 split
        topic = "; ".join(ordered_topics)
        
        return {
            "topic": topic,
            "num_questions": num_questions,
            "difficulty": difficulty,
            "question_type": question_type,
            "question_type_counts": question_type_counts,
            "error_message": None
        }
    
    def retrieve_node(self, state: GraphState) -> Dict:
        """知识库检索节点 - 根据主题检索相关文档"""
        print(f"正在检索主题 '{state['topic']}' 的相关资料...")
        
        try:
            # 检查vectorstore是否已正确加载
            if self.vectorstore is None:
                error_msg = "向量知识库未正确加载，请检查database_agent_mayuan目录是否存在"
                print(error_msg)
                return {
                    "retrieved_docs": [],
                    "error_message": error_msg
                }
            
            # 支持多主题检索
            topic_str = state["topic"]
            topic_list = [t.strip() for t in re.split(r"[;；、，]", topic_str) if t.strip()]

            retrieved_docs: List[str] = []
            for tp in topic_list:
                query = f"{tp} 马克思主义基本原理"
                docs = self.vectorstore.similarity_search(query, k=3)
                retrieved_docs.extend([doc.page_content for doc in docs])

            # 去重并截取前5条
            retrieved_docs = list(dict.fromkeys(retrieved_docs))[:5]

            print(f"成功检索到 {len(retrieved_docs)} 个相关文档片段（共检索主题：{', '.join(topic_list)}）")
            
            return {
                "retrieved_docs": retrieved_docs,
                "error_message": None
            }
            
        except Exception as e:
            error_msg = f"检索过程中出现错误: {e}"
            print(error_msg)
            return {
                "retrieved_docs": [],
                "error_message": error_msg
            }
    
    def generate_node(self, state: GraphState) -> Dict:
        """题目生成节点 - 使用大模型生成题目"""
        print("正在生成题目...")
        
        try:
            # 构建上下文内容
            context = "\n\n".join(state["retrieved_docs"][:3])  # 使用前3个最相关的文档
            
            # 根据题型决定 Prompt 模板

            # ---------- 多题型（混合）处理 ----------
            if state.get("question_type") == "混合":
                # 构造题型及数量描述
                type_details = "\n".join([
                    f"- {qt}：{cnt}道" for qt, cnt in state.get("question_type_counts", {}).items()
                ])

                prompt_template = PromptTemplate.from_template("""
你是一位资深的马克思主义基本原理课程教师，具有丰富的出题经验。请根据以下要求生成高质量的题目。

**任务要求：**
- 主题：{topic}
- 题目类型及数量：
{type_details}
- 难度等级：{difficulty}

**参考资料：**
{context}

**出题要求：**
1. 题目必须严格基于提供的参考资料内容
2. {difficulty}难度的题目特点：
   - 简单：考查基本概念和定义的理解
   - 中等：考查概念间的关系和应用
   - 困难：考查深层理解、分析和综合运用能力
3. 各题型格式要求：
   - 选择题：题干 + 4个选项（A、B、C、D）+ 正确答案 + 简要解析
   - 判断题：题干 + 正确答案（正确/错误）+ 简要解析
   - 材料分析/简答题：题干（可含材料）+ 参考答案 + 简要解析
4. 语言表达要准确、严谨

**输出格式示例：**
选择题1：
题干：[具体题目内容]
A. [选项A]
B. [选项B]
C. [选项C]
D. [选项D]
正确答案：[正确选项]
解析：[简要解析说明]

判断题1：
题干：[具体题目内容]
正确答案：[正确/错误]
解析：[简要解析说明]

简答题1：
题干：[具体题目内容]
参考答案：[答案内容]
解析：[简要解析说明]

请按照题型分类并保持题号连续。
""")

                formatted_prompt = prompt_template.format(
                    topic=state["topic"],
                    type_details=type_details,
                    difficulty=state["difficulty"],
                    context=context
                )
                # 加入原始用户需求，确保题型-主题对应
                formatted_prompt += f"\n\n**原始用户需求：** {state.get('user_input', '')}\n\n请严格按照上述需求生成题目，包括题型、数量及对应主题的匹配。"

            # ---------- 单一题型处理 ----------
            else:
                # 根据题目类型选择相应的模板
                q_type = state["question_type"]

                if q_type == "判断题":
                    prompt_template = PromptTemplate.from_template("""
你是一位资深的马克思主义基本原理课程教师，具有丰富的出题经验。请根据以下要求生成高质量的题目。

**任务要求：**
- 主题：{topic}
- 题目数量：{num_questions}道
- 难度等级：{difficulty}
- 题目类型：判断题

**参考资料：**
{context}

**出题要求：**
1. 题目必须严格基于提供的参考资料内容
2. {difficulty}难度的题目特点同前
3. 判断题格式：题干 + 正确答案（正确/错误）+ 简要解析
4. 语言表达要准确、严谨

**输出格式：**
题目1：
题干：[具体题目内容]
正确答案：[正确/错误]
解析：[简要解析说明]

题目2：
...
""")
                elif q_type == "简答题":
                    prompt_template = PromptTemplate.from_template("""
你是一位资深的马克思主义基本原理课程教师，具有丰富的出题经验。请根据以下要求生成高质量的题目。

**任务要求：**
- 主题：{topic}
- 题目数量：{num_questions}道
- 难度等级：{difficulty}
- 题目类型：材料分析/简答题

**参考资料：**
{context}

**出题要求：**
1. 题目必须严格基于提供的参考资料内容
2. {difficulty}难度的题目特点同前
3. 每道材料分析/简答题包含：题干（可提供材料或问题描述）、参考答案、简要解析
4. 语言表达要准确、严谨

**输出格式：**
题目1：
题干：[具体题目内容]
参考答案：[答案内容]
解析：[简要解析说明]

题目2：
...
""")
                else:  # 默认选择题
                    prompt_template = PromptTemplate.from_template("""
你是一位资深的马克思主义基本原理课程教师，具有丰富的出题经验。请根据以下要求生成高质量的题目。

**任务要求：**
- 主题：{topic}
- 题目数量：{num_questions}道
- 难度等级：{difficulty}
- 题目类型：选择题

**参考资料：**
{context}

**出题要求：**
1. 题目必须严格基于提供的参考资料内容
2. {difficulty}难度的题目特点：
   - 简单：考查基本概念和定义的理解
   - 中等：考查概念间的关系和应用
   - 困难：考查深层理解、分析和综合运用能力
3. 每道选择题包含：题干、4个选项（A、B、C、D）、正确答案和简要解析
4. 选项设计要合理，干扰项要有一定迷惑性
5. 语言表达要准确、严谨

**输出格式：**
题目1：
题干：[具体题目内容]
A. [选项A]
B. [选项B]
C. [选项C]
D. [选项D]
正确答案：[正确选项]
解析：[简要解析说明]

题目2：
...
""")

                formatted_prompt = prompt_template.format(
                    topic=state["topic"],
                    num_questions=state["num_questions"],
                    difficulty=state["difficulty"],
                    context=context
                )
                # 加入原始用户需求，确保题型-主题对应
                formatted_prompt += f"\n\n**原始用户需求：** {state.get('user_input', '')}\n\n请严格按照上述需求生成题目，包括题型、数量及对应主题的匹配。"

            # 调用大模型生成题目
            messages = [
                SystemMessage(content="你是一位专业的马克思主义基本原理课程教师，擅长出题和教学。"),
                HumanMessage(content=formatted_prompt)
            ]
            
            response = self.llm.invoke(messages)
            generated_questions = response.content
            
            print("题目生成完成！")
            
            return {
                "generated_questions": generated_questions,
                "error_message": None
            }
            
        except Exception as e:
            error_msg = f"题目生成过程中出现错误: {e}"
            print(error_msg)
            return {
                "generated_questions": "",
                "error_message": error_msg
            }
    
    def _build_graph(self):
        """构建 LangGraph 工作流"""
        print("正在构建工作流图...")
        
        try:
            # 创建状态图
            workflow = StateGraph(GraphState)
            
            # 添加节点
            workflow.add_node("parse_input", self.parse_input_node)
            workflow.add_node("retrieve", self.retrieve_node)
            workflow.add_node("generate", self.generate_node)
            
            # 定义流转关系
            workflow.set_entry_point("parse_input")  # 设置入口点
            workflow.add_edge("parse_input", "retrieve")  # 解析输入 -> 检索
            workflow.add_edge("retrieve", "generate")     # 检索 -> 生成
            workflow.add_edge("generate", END)            # 生成 -> 结束
            
            # 编译图
            self.graph = workflow.compile()
            print("工作流图构建完成！")
            
        except Exception as e:
            print(f"构建工作流图失败: {e}")
            print("Agent将无法正常工作")
            self.graph = None
    
    def process_request(self, user_input: str) -> str:
        """处理用户请求"""
        print(f"\n收到用户请求: {user_input}")
        print("=" * 50)
        
        # 检查graph是否已正确构建
        if self.graph is None:
            return "工作流图未正确构建，请检查系统初始化"
        
        # 初始化状态 - 确保类型符合GraphState
        initial_state: GraphState = {
            "user_input": user_input,
            "topic": "",
            "num_questions": 0,
            "difficulty": "",
            "question_type": "",
            "question_type_counts": {},
            "retrieved_docs": [],
            "generated_questions": "",
            "error_message": None
        }
        
        try:
            # 运行工作流
            final_state = self.graph.invoke(initial_state)
            
            # 检查是否有错误
            if final_state["error_message"]:
                return f"处理过程中出现错误: {final_state['error_message']}"
            
            # 返回生成的题目
            return final_state["generated_questions"]
            
        except Exception as e:
            return f"系统错误: {e}"


def main():
    """主程序入口 - 提供命令行交互界面"""
    print("=" * 60)
    print("   马克思主义基本原理智能出题 Agent")
    print("   基于 LangGraph 和 LangChain 构建")
    print("=" * 60)
    
    # 检查环境变量
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("\n❌ 错误：未设置 DASHSCOPE_API_KEY 环境变量")
        print("\n🔧 请按以下步骤设置 API Key：")
        print("1. 在终端中运行：")
        print("   export DASHSCOPE_API_KEY='your_api_key_here'  # Linux/Mac")
        print("   或")
        print("   $env:DASHSCOPE_API_KEY='your_api_key_here'   # Windows PowerShell")
        print("\n2. 或者在代码开头添加：")
        print("   os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'")
        print("\n💡 您可以在阿里云控制台获取 API Key")
        return
    
    try:
        # 初始化 Agent
        agent = MayuanQuestionAgent()
        print("\n🎉 Agent 初始化完成！")
        
        print("\n💡 使用说明:")
        print("   请输入您的出题需求，例如：")
        print("   - '请给我出5道关于唯物辩证法的中等难度选择题'")
        print("   - '出3道关于实践观的简单题目'")
        print("   - '给我来10道马克思主义哲学困难选择题'")
        print("   - 输入 'quit' 或 'exit' 退出程序")
        print("=" * 60)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n请输入您的出题需求: ").strip()
                
                # 检查退出指令
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("\n👋 谢谢使用，再见！")
                    break
                
                # 检查空输入
                if not user_input:
                    print("⚠️  请输入有效的出题需求")
                    continue
                
                # 处理请求
                result = agent.process_request(user_input)
                
                # 输出结果
                print("\n" + "=" * 50)
                print("📝 生成的题目:")
                print("=" * 50)
                print(result)
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 处理请求时出现错误: {e}")
                print("请重试或输入新的请求")
    
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        print("请检查:")
        print("1. DASHSCOPE_API_KEY 环境变量是否设置正确")
        print("2. 向量数据库文件是否存在于 'database_agent_mayuan' 目录")
        print("3. 网络连接是否正常")


if __name__ == "__main__":
    main() 