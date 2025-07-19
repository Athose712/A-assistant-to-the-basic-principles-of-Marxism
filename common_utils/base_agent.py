"""
This file defines the BaseAgent, which provides a reusable framework for building
subject-specific question-generation agents. It encapsulates the common logic
for parsing user input, retrieving information, generating questions, and managing
the workflow with LangGraph.

To create a new agent (e.g., for Mao's Thoughts), you would:
1. Subclass BaseAgent.
2. Override the `__init__` method to provide subject-specific details like:
   - subject_name: The name of the course (e.g., "毛概").
   - default_topic: A fallback topic if none is detected.
   - common_topics: A list of keywords to identify topics.
   - vectorstore_path: The path to the specialized vector database.
3. The core logic for processing requests is inherited and reused.
"""
import os
import re
from typing import Dict, List, TypedDict, Optional

from langchain_community.vectorstores import FAISS
from langchain_dashscope.embeddings import DashScopeEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel
from langchain_core.messages import HumanMessage, SystemMessage

from .llm_wrapper import CustomChatDashScope
from .prompts import (
    SINGLE_TYPE_PROMPT_TEMPLATE,
    QUESTION_TYPE_CONFIG,
    MIXED_TYPE_PROMPT_TEMPLATE,
    DIFFICULTY_ADDENDUM_HARD,
)

class GraphState(TypedDict):
    """Defines the state structure for the LangGraph workflow."""
    user_input: str
    subject_name: str
    topic: str
    num_questions: int
    difficulty: str
    question_type: str
    question_type_counts: Dict[str, int]
    retrieved_docs: List[str]
    generated_questions: str
    error_message: Optional[str]


class BaseAgent:
    """A base class for creating intelligent question-generation agents."""

    def __init__(
        self,
        subject_name: str,
        default_topic: str,
        common_topics: List[str],
        vectorstore_path: str,
        llm_model: str = "qwen-max",
        embedding_model: str = "text-embedding-v2",
    ):
        """
        Initializes the agent with subject-specific configurations.

        Args:
            subject_name: The display name of the subject (e.g., "马克思主义基本原理").
            default_topic: The default topic to use if none can be parsed.
            common_topics: A list of common topics for quick matching.
            vectorstore_path: Path to the local FAISS vector store.
            llm_model: The LLM model to use for generation.
            embedding_model: The embedding model to use for retrieval.
        """
        self.subject_name = subject_name
        self.default_topic = default_topic
        self.common_topics = common_topics
        self.vectorstore_path = vectorstore_path
        
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise ValueError("DASHSCOPE_API_KEY environment variable not set.")

        try:
            self.embeddings = DashScopeEmbeddings(model=embedding_model)
            self.llm = CustomChatDashScope(model=llm_model, temperature=0.7)
            print(f"[{self.subject_name}] LLM and Embedding models initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {e}")

        self.vectorstore = self._load_knowledge_base()
        self.graph: Pregel = self._build_graph()

    def _load_knowledge_base(self) -> Optional[FAISS]:
        """Loads the vector knowledge base from the specified path."""
        try:
            print(f"[{self.subject_name}] Loading knowledge base from '{self.vectorstore_path}'...")
            return FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(f"Warning: Failed to load knowledge base: {e}. Agent will run without retrieval.")
            return None

    def _build_graph(self) -> Pregel:
        """Builds the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        workflow.add_node("parse_input", self.parse_input_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)

        workflow.set_entry_point("parse_input")
        workflow.add_edge("parse_input", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        print(f"[{self.subject_name}] Workflow graph built successfully.")
        return workflow.compile()

    def parse_input_node(self, state: GraphState) -> Dict:
        """Parses the user's raw input to extract structured parameters."""
        print(f"[{self.subject_name}] Parsing user input...")
        user_input = state["user_input"]

        # --- Extract Number and Type of Questions ---
        type_count_pattern = r"(\d+)\s*(?:道|题|个)[^\u4e00-\u9fa5]*?(选择题|判断题|简答题|材料\s*分析题?)"
        compact_input = re.sub(r"\s+", "", user_input)
        mixed_matches = re.findall(type_count_pattern, compact_input)
        
        question_type_counts = {}
        for num_str, q_type_raw in mixed_matches:
            q_type_normalized = q_type_raw.replace("材料分析题", "简答题").replace("材料分析", "简答题")
            question_type_counts[q_type_normalized] = question_type_counts.get(q_type_normalized, 0) + int(num_str)

        if question_type_counts:
            num_questions = sum(question_type_counts.values())
        else:
            num_pattern_simple = r"(\d+)\s*(?:道|题|个)"
            numbers = re.findall(num_pattern_simple, user_input)
            num_questions = sum(int(n) for n in numbers) if numbers else 5

        # --- Extract Difficulty ---
        difficulty = "中等"
        if any(kw in user_input for kw in ["简单", "容易", "基础"]):
            difficulty = "简单"
        elif any(kw in user_input for kw in ["困难", "难", "高级"]):
            difficulty = "困难"

        # --- Determine Question Type(s) ---
        detected_types = [qt for qt in ["选择题", "判断题", "简答题"] if qt in user_input or ("简答题" == qt and "材料分析" in user_input)]
        detected_types = list(dict.fromkeys(detected_types))

        if not question_type_counts and detected_types:
            avg_count = max(1, num_questions // len(detected_types))
            for qt in detected_types:
                question_type_counts[qt] = avg_count
        
        if len(question_type_counts) > 1:
            question_type = "混合"
        elif len(question_type_counts) == 1:
            question_type = next(iter(question_type_counts))
        else:
            question_type = "选择题"
            question_type_counts = {"选择题": num_questions}

        # --- Extract Topic ---
        detected_topics = [t for t in self.common_topics if t in user_input]
        about_matches = re.findall(r"关于(.*?)的", user_input)
        for raw in about_matches:
            cleaned = re.sub(r"(简单|容易|基础|中等|困难|高级|选择题|判断题|简答题|材料\s*分析题?|题目|\s)", "", raw).strip()
            if cleaned:
                detected_topics.append(cleaned)
        
        ordered_topics = list(dict.fromkeys(detected_topics))
        if not ordered_topics:
            fallback_topic = re.sub(r'\d+道|\d+题|请|给我|出|关于|的|简单|中等|困难|选择题|判断题|简答题|材料\s*分析题?|题目', '', user_input).strip()
            ordered_topics = [fallback_topic if fallback_topic else self.default_topic]

        return {
            "topic": "; ".join(ordered_topics),
            "num_questions": num_questions,
            "difficulty": difficulty,
            "question_type": question_type,
            "question_type_counts": question_type_counts,
            "subject_name": self.subject_name,
            "error_message": None,
        }

    def retrieve_node(self, state: GraphState) -> Dict:
        """Retrieves relevant documents from the knowledge base."""
        print(f"[{self.subject_name}] Retrieving documents for topic: '{state['topic']}'...")
        if self.vectorstore is None:
            return {"retrieved_docs": [], "error_message": "Knowledge base not loaded."}

        try:
            topic_list = [t.strip() for t in re.split(r"[;；、，]", state["topic"]) if t.strip()]
            retrieved_docs = []
            for tp in topic_list:
                docs = self.vectorstore.similarity_search(f"{tp} {self.subject_name}", k=3)
                retrieved_docs.extend([doc.page_content for doc in docs])
            
            unique_docs = list(dict.fromkeys(retrieved_docs))[:5]
            print(f"[{self.subject_name}] Retrieved {len(unique_docs)} unique document snippets.")
            return {"retrieved_docs": unique_docs, "error_message": None}
        except Exception as e:
            return {"retrieved_docs": [], "error_message": f"Retrieval failed: {e}"}

    def generate_node(self, state: GraphState) -> Dict:
        """Generates questions using the LLM based on the retrieved context."""
        print(f"[{self.subject_name}] Generating questions...")
        context = "\n\n".join(state["retrieved_docs"][:3])
        
        try:
            if state["question_type"] == "混合":
                type_details = "\n".join([f"- {qt}：{cnt}道" for qt, cnt in state["question_type_counts"].items()])
                prompt = MIXED_TYPE_PROMPT_TEMPLATE.format(
                    subject_name=self.subject_name,
                    topic=state["topic"],
                    type_details=type_details,
                    difficulty=state["difficulty"],
                    context=context,
                    user_input=state["user_input"],
                )
            else:
                q_type = state["question_type"]
                config = QUESTION_TYPE_CONFIG.get(q_type, QUESTION_TYPE_CONFIG["选择题"])
                prompt = SINGLE_TYPE_PROMPT_TEMPLATE.format(
                    subject_name=self.subject_name,
                    topic=state["topic"],
                    num_questions=state["num_questions"],
                    difficulty=state["difficulty"],
                    question_type_specific_name=config["question_type_specific_name"],
                    format_requirements=config["format_requirements"],
                    output_format_example=config["output_format_example"],
                    context=context,
                    user_input=state["user_input"],
                )

            if state["difficulty"] == "困难":
                prompt += f"\n\n{DIFFICULTY_ADDENDUM_HARD}"
            
            messages = [
                SystemMessage(content=f"你是一位专业的{self.subject_name}课程教师，擅长出题和教学。"),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            
            print(f"[{self.subject_name}] Question generation complete.")
            return {"generated_questions": response.content, "error_message": None}
        except Exception as e:
            return {"generated_questions": "", "error_message": f"Generation failed: {e}"}

    def process_request(self, user_input: str) -> str:
        """Processes a user's request through the entire workflow."""
        if not self.graph:
            return "Error: Agent graph is not compiled."
        
        initial_state = GraphState(
            user_input=user_input,
            subject_name=self.subject_name,
            topic="",
            num_questions=0,
            difficulty="",
            question_type="",
            question_type_counts={},
            retrieved_docs=[],
            generated_questions="",
            error_message=None,
        )
        
        try:
            final_state = self.graph.invoke(initial_state)
            if final_state["error_message"]:
                return f"An error occurred: {final_state['error_message']}"
            return final_state["generated_questions"]
        except Exception as e:
            return f"A system error occurred: {e}" 