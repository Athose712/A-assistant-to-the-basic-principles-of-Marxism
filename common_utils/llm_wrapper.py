import os
from typing import Any, List, Optional, Union
import base64

import dashscope
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

import logging

# Set up API key for DashScope SDK
api_key = os.environ.get("DASHSCOPE_API_KEY")
if api_key:
    dashscope.api_key = api_key

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CustomChatDashScope(BaseChatModel):
    """A stable DashScope chat model wrapper implementing LangChain's BaseChatModel.

    This class is extracted into *common_utils* so that multiple agents (e.g. 毛概、习概)
    can import the same implementation without duplication.
    """

    model: str = "qwen-turbo"
    temperature: float = 0.7

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Sends messages to DashScope and returns the response as AIMessage."""

        prompt_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                prompt_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                prompt_messages.append({"role": "assistant", "content": msg.content})

        response = dashscope.Generation.call(
            model=self.model,
            messages=prompt_messages,
            result_format="message",
            temperature=self.temperature,
            stream=False,
            **kwargs,
        )

        # Non-streaming mode -> GenerationResponse with status_code / output
        if hasattr(response, "status_code"):
            if response.status_code == 200:  # type: ignore[attr-defined]
                ai_content = response.output.choices[0]["message"]["content"]  # type: ignore[attr-defined]
                return AIMessage(content=ai_content)
        raise Exception(
                "DashScope API Error: Code {} , Message {}".format(  # type: ignore[attr-defined]
                    getattr(response, "code", "unknown"), getattr(response, "message", "unknown")
                )
            )

        # Streaming mode -> concatenate chunks
        if hasattr(response, "__iter__"):
            content_chunks: List[str] = []
            for chunk in response:
                try:
                    content_chunks.append(chunk.choices[0].delta.content)
                except Exception:
                    pass  # Some chunks may not have delta.content
            return AIMessage(content="".join(content_chunks))

        raise Exception("DashScope returned an unknown response type and could not be parsed.")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        ai_msg = self._call(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    # ---------------------------------------------------------------------
    # LangChain required properties
    # ---------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:  # noqa: D401 – keeping LangChain naming convention
        return "custom_chat_dashscope_wrapper"


class CustomVisionChatDashScope(BaseChatModel):
    """A DashScope chat model wrapper with vision capabilities for multi-modal inputs.
    
    This class extends the base functionality to support image analysis using
    DashScope's vision-language models (e.g., qwen-vl-max).
    """

    model: str = "qwen-vl-max"
    temperature: float = 0.7

    def _encode_image_base64(self, image_path: str) -> str:
        """将图片文件编码为base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _prepare_multimodal_content(self, text: str, image_path: Optional[str] = None) -> List[dict]:
        """准备多模态内容，支持文本和图片

        关键改进:
        1. 动态检测图片 MIME 类型，确保前缀与实际格式一致
        2. 对过大的图片进行等比缩放到最长边不超过 1024px，避免模型处理失败
        3. 对任何异常情况降级为仅文本输入而不会直接抛错
        """
        content: List[dict] = []

        # 添加文本部分
        if text:
            content.append({"text": text})

        # 如果没有图片，直接返回
        if not image_path:
            return content

        # --------------------------------------------------
        # 处理图片
        # --------------------------------------------------
        # 1. data:image;base64 直接透传
        if image_path.startswith("data:image"):
            content.append({"image": image_path})
            return content

        # 2. 本地文件路径，需要进行编码并增加 MIME 前缀
        try:
            import mimetypes
            import io
            from PIL import Image

            # 检测 MIME 类型，若失败则默认 jpeg
            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type is None:
                mime_type = "image/jpeg"

            # 打开图片并在必要时缩放
            with Image.open(image_path) as img:
                max_dim = 1024  # DashScope 推荐最长边不超过 1024
                if max(img.size) > max_dim:
                    ratio = max_dim / float(max(img.size))
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size)

                buffered = io.BytesIO()
                # 推断保存格式（JPEG/PNG/WebP/...）
                save_format = mime_type.split("/")[-1].upper()
                if save_format == "JPG":
                    save_format = "JPEG"
                img.save(buffered, format=save_format)
                encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            content.append({"image": f"data:{mime_type};base64,{encoded_image}"})
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            content.append({"text": f"[图片加载失败: {str(e)}]"})

        return content

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        image_path: Optional[str] = None,  # 现在image_path可以用于任何消息，但我们需要调整逻辑
        **kwargs: Any,
    ) -> AIMessage:
        """调用DashScope Vision API处理多模态输入"""

        prompt_messages = []
        image_added = False  # 跟踪是否已添加图像，以避免重复
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                content = msg.content
                if image_path and not image_added:
                    # 为带有图像的用户消息准备多模态内容
                    multimodal_content = self._prepare_multimodal_content(content, image_path)
                    prompt_messages.append({"role": "user", "content": multimodal_content})
                    image_added = True
                else:
                    prompt_messages.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                prompt_messages.append({"role": "assistant", "content": msg.content})

        try:
            response = dashscope.MultiModalConversation.call(
                model=self.model,
                messages=prompt_messages,
                temperature=self.temperature,
                timeout=30,  # 添加超时（秒）
                **kwargs,
            )

            # 处理API响应
            if hasattr(response, "status_code"):
                if response.status_code == 200:
                    ai_content = response.output.choices[0]["message"]["content"]
                    return AIMessage(content=ai_content)
                else:
                    error_msg = f"DashScope Vision API Error: Code {response.status_code}"
                    if hasattr(response, 'message'):
                        error_msg += f", Message: {response.message}"
                    raise Exception(error_msg)
            else:
                raise Exception("DashScope Vision API returned unexpected response format")

        except Exception as e:
            logging.error(f"视觉API调用失败: {e}")
            
            # 移除图片内容，只保留文本
            text_messages = []
            for msg_data in prompt_messages:
                if isinstance(msg_data.get("content"), list):
                    # 提取文本内容
                    text_parts = [item.get("text", "") for item in msg_data["content"] if "text" in item]
                    text_content = " ".join(text_parts)
                    text_messages.append({"role": msg_data["role"], "content": text_content})
                else:
                    text_messages.append(msg_data)
            
            # 使用普通文本API
            response = dashscope.Generation.call(
                model="qwen-turbo",
                messages=text_messages,
                result_format="message",
                temperature=self.temperature,
                stream=False,
                timeout=30,  # 添加超时
                **kwargs,
            )
            
            if hasattr(response, "status_code") and response.status_code == 200:
                ai_content = response.output.choices[0]["message"]["content"]
                logging.info("回退到文本模式成功")
                return AIMessage(content=f"[注意：图片分析功能暂时不可用，以下是基于文本的回复]\n\n{ai_content}")
            else:
                raise Exception("文本模式API调用也失败了")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        image_path: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatResult:
        ai_msg = self._call(messages, stop=stop, image_path=image_path, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def call_with_image(
        self,
        text: str,
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """便捷方法：直接调用多模态功能"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=text))
        
        result = self._call(messages, image_path=image_path)
        return result.content

    @property
    def _llm_type(self) -> str:
        return "custom_vision_chat_dashscope_wrapper" 