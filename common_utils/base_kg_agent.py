"""
This file defines the BaseKnowledgeGraphAgent, a reusable framework for
generating Mermaid-format knowledge graphs for any subject.
"""
import os
import re
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_dashscope.embeddings import DashScopeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from .llm_wrapper import CustomChatDashScope


class BaseKnowledgeGraphAgent:
    """Base class for generating Mermaid-format knowledge graphs."""

    def __init__(self, subject_name: str, vectorstore_path: str):
        """
        Initializes the base knowledge graph agent.

        Args:
            subject_name: The name of the subject (e.g., "马克思主义基本原理").
            vectorstore_path: The path to the FAISS vector store.
        """
        self.subject_name = subject_name
        self.vectorstore_path = vectorstore_path

        if "DASHSCOPE_API_KEY" not in os.environ:
            raise EnvironmentError("Please set the DASHSCOPE_API_KEY environment variable.")

        try:
            self.embeddings = DashScopeEmbeddings(model="text-embedding-v2")
            self.llm = CustomChatDashScope(model="qwen-max", temperature=0.5)
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {e}")

        try:
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load vector store from {self.vectorstore_path}: {e}")

        self.graph_prompt = PromptTemplate.from_template(
            """
你是一位{subject_name}知识图谱专家。请利用提供的"参考资料"，围绕知识点"{topic}"构建一个 Mermaid mindmap（思维导图）格式的知识图谱，突出关键概念及其主要关系，并保持简洁易读。

输出要求：
1. mindmap 总节点不超过 15 个，层级不超过 3 级，保证图谱信息清晰、结构美观，便于学生学习和整理思路。
2. 先输出 Mermaid 源代码，必须使用如下代码块格式：
```mermaid
mindmap
  root(({topic}))
    概念1
      子概念A
    概念2
```
3. Mermaid 代码块结束后，换行再输出一段不超过 100 字的中文总结，对图谱内容进行简洁概括。
4. 除以上内容外，不要输出其他文字。

参考资料：
{context}
"""
        )

    def _retrieve_docs(self, topic: str, k: int = 5) -> List[str]:
        """Retrieves relevant document snippets based on the topic."""
        query = f"{topic} {self.subject_name}"
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def _generate_mermaid(self, topic: str, context: str) -> str:
        """Generates Mermaid code using the large language model."""
        prompt_text = self.graph_prompt.format(
            subject_name=self.subject_name, topic=topic, context=context
        )
        messages = [
            SystemMessage(content="你是一位精通知识图谱构建的学者。"),
            HumanMessage(content=prompt_text),
        ]
        response = self.llm.invoke(messages)
        if hasattr(response, 'content'):
            return str(response.content).strip()
        return str(response).strip()

    def _format_mermaid_response(self, raw_output: str) -> str:
        """
        Post-processes the LLM output to ensure the Mermaid code is correctly formatted.
        """
        raw_output = raw_output.strip()

        # Use regex to find the mermaid block and the summary
        mermaid_match = re.search(r"```mermaid(.*?)```", raw_output, re.DOTALL)
        
        if mermaid_match:
            mermaid_code = mermaid_match.group(1).strip()
            # The summary is whatever is left after the mermaid block
            summary = raw_output[mermaid_match.end():].strip()
        else:
            # If no mermaid block is found, assume the whole output is mermaid code
            mermaid_code = raw_output.replace("```mermaid", "").replace("```", "")
            summary = ""

        # Reconstruct the output in a standard format
        formatted_output = f"```mermaid\n{mermaid_code}\n```"
        if summary:
            formatted_output += f"\n\n{summary}"
            
        return formatted_output.strip()

    def build_knowledge_graph(self, topic: str) -> str:
        """
        Main workflow: retrieves context -> generates Mermaid code and summary.
        """
        docs = self._retrieve_docs(topic, k=5)
        context = "\n\n".join(docs)
        raw_output = self._generate_mermaid(topic, context)
        return self._format_mermaid_response(raw_output) 