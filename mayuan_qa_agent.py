from __future__ import annotations

"""
MayuanAnswerAgent
=================

An intelligent Q&A agent for the course "马克思主义基本原理" (Basic Principles of Marxism).

The agent answers conceptual questions raised by the user. It retrieves relevant
snippets from the subject-specific vector store and then asks the LLM to craft a
concise, accurate answer in Chinese.

The implementation purposefully mirrors the structure used by the existing
question-generation (BaseAgent) and knowledge-graph agents so that all three can
be easily orchestrated together in *app.py*.
"""

import os
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from common_utils.base_retrieval_agent import BaseRetrievalAgent
from common_utils.multimodal_agent import MayuanMultimodalAgent


class MayuanAnswerAgent(BaseRetrievalAgent):
    """Retrieval-augmented Q&A agent with optional image support."""

    def __init__(
        self,
        *,
        subject_name: str = "马克思主义基本原理",
        vectorstore_path: str = "database_agent_mayuan",
        llm_model: str = "qwen-max",
        temperature: float = 0.3,
        embedding_model: str = "text-embedding-v2",
    ) -> None:

        # Initialise retrieval/LLM via base class
        super().__init__(
            subject_name=subject_name,
            vectorstore_path=vectorstore_path,
            llm_model=llm_model,
            embedding_model=embedding_model,
            temperature=temperature,
        )

        # ----- Multimodal capability -----------------------------------------
        try:
            self.multimodal_agent = MayuanMultimodalAgent()
            print(f"[{self.subject_name}] AnswerAgent – multimodal support initialised.")
        except Exception as exc:
            print(f"[{self.subject_name}] ⚠️  Multimodal initialisation failed: {exc}")
            self.multimodal_agent = None

    # ---------------------------------------------------------------------
    # Multimodal interface
    # ---------------------------------------------------------------------
    def process_multimodal_request(self, text_input: str, image_path: Optional[str] = None) -> str:
        """Handle user question that includes an image.

        • If *image_path* is None or multimodal agent unavailable → fallback to text-only *process_request*.
        • Otherwise, call *MayuanMultimodalAgent* to analyse the picture, then post-process/shorten the answer.
        """
        # 没有图片时直接走文本
        if not image_path:
            return self.process_request(text_input)

        # ---------- OCR 预提取文本 ----------
        extracted_text = self._extract_text_from_image(image_path)

        # 如果多模态模型不可用，则尽量用 OCR 文本辅助回答
        if self.multimodal_agent is None:
            if extracted_text:
                base = text_input or "请根据图片中的题目进行解答。"
                combined = base + "\n\n以下是 OCR 自动识别的图片文字，请据此解答：\n" + extracted_text[:1500]
                return self.process_request(combined)
            # 无法 OCR 时退回原始文本
            return self.process_request(text_input)

        # 多模态模型可用 → 文本+图片联合分析
        combined_input = text_input
        if extracted_text:
            combined_input += "\n\n以下是 OCR 自动识别的图片文字，请据此解答：\n" + extracted_text[:1500]

        try:
            response = self.multimodal_agent.process_multimodal_request(combined_input, image_path)
            # 保留原有的段落与列表等 Markdown 格式
            return response.strip()
        except Exception as exc:
            print(f"[{self.subject_name}] Multimodal processing failed, fallback to text: {exc}")
            # 若多模态失败，尝试仅用 OCR 文字回答
            if extracted_text:
                return self.process_request(extracted_text)
            return self.process_request(text_input)

    # ---------------------------------------------------------------------
    # Text-only interface
    # ---------------------------------------------------------------------
    def _build_prompt(self, user_question: str, context: str) -> str:
        """Assemble the prompt sent to the LLM with structured Markdown formatting guidance."""
        return (
            f"你是一位精通{self.subject_name}的教师，请以结构化 Markdown 输出，条理清晰、重点明确。"
            "请优先完整与准确，适度展开，避免冗余。若下方\"参考资料\"足够，请严格基于资料作答；否则结合你的专业知识回答。\n\n"
            "请按照以下结构组织你的回答（如不适用可省略某些小节）：\n"
            "### 核心结论\n"
            "- 用1-2句加粗给出直接答案或观点。\n\n"
            "### 关键要点\n"
            "- 3-6条要点，每条不超过两句，必要处使用**加粗关键词**。\n\n"
            "### 依据与推理\n"
            "- 用2-5句解释你的论证链条，可引用资料中的关键句（简洁转述）。\n\n"
            "### 示例或应用（可选）\n"
            "- 给出1个贴近教学的例子或场景来帮助理解。\n\n"
            "### 小结与延伸（可选）\n"
            "- 用1-2句总结，并提出1个进一步思考方向。\n\n"
            "参考资料（可能为空）：\n" + context + "\n\n学生问题：" + user_question + "\n回答："
        )

    def process_request(self, user_question: str) -> str:
        """Answer *user_question* using retrieval-augmented generation."""
        # 1. Retrieve docs -----------------------------------------------------
        docs = self._retrieve_docs(f"{user_question} {self.subject_name}", k=5)
        context = "\n\n".join(docs[:5])

        # 2. Build prompt & call LLM -----------------------------------------
        prompt = self._build_prompt(user_question, context)

        messages = [
            SystemMessage(content=(
                "你是一位严谨的马克思主义原理解答专家。"
                "请使用结构化 Markdown（标题、列表、加粗）输出，层次清晰，美观易读；"
                "在保证准确性的前提下适度展开，覆盖关键要点。"
            )),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages, **self.generation_kwargs)
            answer = str(getattr(response, "content", response)).strip()
            answer = re.sub(r"^`+|`+$", "", answer).strip()
            return answer
        except Exception as exc:
            print(f"[{self.subject_name}] Answer generation failed: {exc}")
            return "抱歉，回答过程中出现问题，请稍后再试。"

    # ------------------------------------------------------------------
    def _extract_text_from_image(self, image_path: str) -> str:
        """Simple OCR using pytesseract (if available). Returns empty string on failure."""
        try:
            import os
            import pytesseract
            from PIL import Image
            # 优先使用环境变量中配置的 tesseract 路径，否则尝试常见的 Windows 安装位置
            custom_cmd = os.environ.get("TESSERACT_CMD")
            if custom_cmd and os.path.exists(custom_cmd):
                pytesseract.pytesseract.tesseract_cmd = custom_cmd  # type: ignore[attr-defined]
            else:
                common_paths = [
                    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                    r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
                ]
                for p in common_paths:
                    if os.path.exists(p):
                        pytesseract.pytesseract.tesseract_cmd = p  # type: ignore[attr-defined]
                        break
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang="chi_sim+eng")
            return text.strip()
        except Exception as _:
            return ""
