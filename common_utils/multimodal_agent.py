"""
多模态Agent基类，支持图片和文本输入的AI助手
"""
import os
from typing import Optional, Dict, Any
from .llm_wrapper import CustomVisionChatDashScope

class MultimodalAgent:
    """
    支持多模态输入的AI助手基类
    可以处理文本和图片的组合输入，并给出相应回复
    """

    def __init__(self, subject_name: str = "AI助手", model: str = "qwen-vl-max"):
        """
        初始化多模态Agent
        
        Args:
            subject_name: 学科名称或助手名称
            model: 使用的视觉语言模型
        """
        self.subject_name = subject_name
        self.model = model
        
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise ValueError("DASHSCOPE_API_KEY environment variable not set.")
        
        try:
            self.vision_llm = CustomVisionChatDashScope(model=model, temperature=0.7)
            print(f"[{self.subject_name}] 多模态AI助手初始化成功")
        except Exception as e:
            raise RuntimeError(f"多模态模型初始化失败: {e}")

    def process_multimodal_request(
        self, 
        text_input: str, 
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        处理多模态请求（文本+图片）
        
        Args:
            text_input: 用户的文本输入
            image_path: 图片文件路径（可选）
            system_prompt: 系统提示词（可选）
            
        Returns:
            AI助手的回复
        """
        try:
            # 如果没有提供系统提示词，使用默认的
            if not system_prompt:
                system_prompt = self._get_default_system_prompt()
            
            # 调用视觉语言模型
            response = self.vision_llm.call_with_image(
                text=text_input,
                image_path=image_path,
                system_prompt=system_prompt
            )
            
            return response
            
        except Exception as e:
            error_msg = f"处理多模态请求时出错: {str(e)}"
            print(f"[{self.subject_name}] {error_msg}")
            return f"抱歉，{error_msg}。请重试或检查您的输入。"

    def _get_default_system_prompt(self) -> str:
        """获取默认的系统提示词"""
        return f"""你是一个专业的{self.subject_name}AI助手。你能够理解和分析用户提供的文本和图片内容。

请遵循以下原则：
1. 如果用户提供了图片，请仔细分析图片内容
2. 结合图片信息和文本问题，给出专业、准确的回答
3. 如果图片与{self.subject_name}相关，请结合专业知识进行分析
4. 保持回答的清晰性和逻辑性
5. 如果无法识别图片内容，请诚实说明
6. 如果图片与主题无关，请礼貌地重定向到相关话题讨论

请用中文回答用户的问题。"""

class MayuanMultimodalAgent(MultimodalAgent):
    """马克思主义基本原理多模态Agent"""
    
    def __init__(self):
        super().__init__(subject_name="马克思主义基本原理", model="qwen-vl-max")
    
    def _get_default_system_prompt(self) -> str:
        """获取马原专业的系统提示词"""
        return """你是一个专业的马克思主义基本原理AI助手。你能够理解和分析用户提供的文本和图片内容。

请遵循以下原则：
1. 如果用户提供了图片，请仔细分析图片内容，特别关注与马克思主义理论相关的内容
2. 结合图片信息和用户问题，从马克思主义基本原理的角度进行分析和回答
3. 如果图片中包含哲学概念、理论图表、名人肖像等，请结合马克思主义理论进行专业解读
4. 可以分析图片中的文字、图表、概念图等，并用马克思主义理论进行阐释
5. 保持回答的理论性、准确性和教育性
6. 如果图片内容与马克思主义无关，也请客观分析，但尽量引导到相关理论思考
7. 如果图片与当前主题无关，请礼貌地建议用户上传相关图片或重定向讨论

涉及的主要内容包括：
- 唯物辩证法（对立统一、质量互变、否定之否定）
- 历史唯物主义
- 马克思主义认识论
- 马克思主义政治经济学基本原理
- 科学社会主义基本原理

请用中文回答用户的问题，确保回答专业、准确、有教育意义。"""

class SocratesMultimodalAgent(MultimodalAgent):
    """苏格拉底式对话多模态Agent"""
    
    def __init__(self, character: str = "马克思", topic: str = "马克思主义理论"):
        super().__init__(subject_name="历史思想家对话", model="qwen-vl-max")
        self.character = character
        self.current_topic = topic
    
    def update_dialogue_context(self, character: str, topic: str):
        """更新对话上下文"""
        self.character = character
        self.current_topic = topic
    
    def _get_default_system_prompt(self) -> str:
        """获取苏格拉底式对话的系统提示词"""
        return f"""你现在要扮演{self.character}，与用户进行苏格拉底式对话，探讨"{self.current_topic}"这个主题。

如果用户提供了图片，请：
1. 仔细分析图片内容，包括文字、图表、人物等
2. 以{self.character}的身份和观点来理解和解读图片
3. 结合图片内容，用苏格拉底式的方法引导用户思考
4. 如果图片与讨论主题相关，深入分析其理论意义
5. 如果图片与主题无关，请礼貌地重定向到当前话题

苏格拉底式对话的特点：
- 通过提问引导用户思考，而不是直接给出答案
- 挖掘用户观点中的假设和逻辑问题
- 循序渐进地引导用户发现真理
- 保持{self.character}的语言风格和理论背景

请用中文进行对话，体现{self.character}的思想特色和对话风格。"""