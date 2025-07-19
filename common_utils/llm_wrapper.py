import os
from typing import Any, List, Optional

import dashscope
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

# Set up API key for DashScope SDK
api_key = os.environ.get("DASHSCOPE_API_KEY")
if api_key:
    dashscope.api_key = api_key


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