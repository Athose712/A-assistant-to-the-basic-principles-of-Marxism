"""
马克思主义基本原理智能出题 Agent
"""
import os
from typing import Optional
from common_utils.base_agent import BaseAgent
from common_utils.multimodal_agent import MayuanMultimodalAgent

class MayuanQuestionAgent(BaseAgent):
    """
    专门用于“马克思主义基本原理”课程的智能出题 Agent。
    它继承自 BaseAgent，并提供了该课程特有的配置。
    """
    def __init__(self):
        """初始化马原 Agent 的特定配置"""
        
        # 定义马原课程的常见主题，用于更精确地解析用户输入
        common_topics = [
            "唯物辩证法", "历史唯物主义", "马克思主义哲学", "认识论",
            "实践观", "矛盾论", "否定之否定", "质量互变", "联系",
            "发展", "本质与现象", "内容与形式", "原因与结果",
            "必然与偶然", "可能与现实", "社会存在", "社会意识",
            "辩证唯物主义"
        ]

        # 调用父类的构造函数，传入马原课程的特定参数
        super().__init__(
            subject_name="马克思主义基本原理",
            default_topic="马克思主义基本原理",
            common_topics=common_topics,
            vectorstore_path="database_agent_mayuan"
        )
        
        # 初始化多模态Agent用于图片分析
        try:
            self.multimodal_agent = MayuanMultimodalAgent()
            print("[马原Agent] 多模态功能初始化成功")
        except Exception as e:
            print(f"[马原Agent] 多模态功能初始化失败: {e}")
            self.multimodal_agent = None

        # --------------------------------------------------
        # 状态缓存：保存最近一次生成的题目（含答案解析）
        # 以及对应的仅题干版本，便于后续按需提供解析
        # --------------------------------------------------
        self._last_full_output: str = ""
        self._last_question_only_output: str = ""
    
    # --------------------------------------------------
    # 公共接口
    # --------------------------------------------------
    def process_request(self, user_input: str) -> str:
        """重写父类方法，以支持“按需提供解析”的逻辑"""
        # 如果用户明确索要解析/答案，则直接返回上一次的完整内容
        if any(kw in user_input for kw in ["解析", "答案", "讲解", "答案解析", "参考答案"]):
            if self._last_full_output:
                return self._last_full_output
            # 若没有缓存，则提示用户先生成题目
            return "当前没有可供解析的题目，请先提出出题需求。"

        # 否则视为新的出题需求，调用父类生成题目
        full_output = super().process_request(user_input)

        # 生成后保存完整内容
        self._last_full_output = full_output
        # 同时生成“去除正确答案/解析”的版本
        self._last_question_only_output = self._strip_explanations(full_output)
        return self._last_question_only_output

    # --------------------------------------------------
    # 多模态接口保持不变，内部仍会回退到 process_request
    # --------------------------------------------------
    def process_multimodal_request(self, text_input: str, image_path: Optional[str] = None) -> str:
        """
        处理多模态请求（支持图片+文本输入）
        """
        # 若未提供图片或多模态未初始化，则退回文本出题流程（保留按需解析逻辑）
        if not image_path or not self.multimodal_agent:
            return self.process_request(text_input)

        # 如果用户这次是来“索要解析/答案”，优先返回缓存的完整内容
        if any(kw in text_input for kw in ["解析", "答案", "讲解", "答案解析", "参考答案"]):
            return self._last_full_output or "当前没有可供解析的题目，请先提出出题需求。"

        try:
            # 走多模态模型生成完整内容
            full_output = self.multimodal_agent.process_multimodal_request(text_input, image_path)
            # 将完整内容纳入缓存
            self._last_full_output = full_output

            # 如果本次请求属于出题场景（包含常见出题关键词），则先隐藏答案/解析
            if any(kw in text_input for kw in ["出题", "生成题目", "题目", "选择题", "判断题", "简答题", "试题", "练习"]):
                self._last_question_only_output = self._strip_explanations(full_output)
                return self._last_question_only_output

            # 否则按多模态原样返回
            return full_output
        except Exception as e:
            print(f"[马原Agent] 多模态处理失败，回退到文本模式: {e}")
            return self.process_request(text_input)

    # --------------------------------------------------
    # 私有工具方法
    # --------------------------------------------------
    def _strip_explanations(self, text: str) -> str:
        """移除答案与解析，只保留题干及选项/题号。

        规则改进：
        - 识别更丰富的“答案/解析”起始样式（如：正确答案、参考答案、标准答案、答案是/为、答：、解：、【答案】等）。
        - 支持“解析：”换行后的多行内容整段剥离，直到检测到下一题/新段落标题为止。
        - 处理同一行内含有“答案：B”之类的内嵌写法（整行移除）。
        """
        import re

        lines = text.splitlines()
        filtered: list[str] = []

        # 起始信号（命中后进入剥离块）
        start_patterns = [
            r"^\s*(?:正确?答案|参考答案|标准答案|答案解析|解析|解答|讲解|评分标准|思路|分析|参考思路|答案是|答案为|Answer|Explanation)\s*[:：】\])]?.*$",
            r"^\s*[（(【\[]?(?:答|解)\s*[：:]\s*.*$",
            r"^\s*[【\[]?(?:答案|解析|参考答案)[】\]]\s*.*$",
        ]

        # 内嵌写法（行中出现“答案/解析”也视为需要移除整行）
        inline_patterns = [
            r"(正确?答案|参考答案|标准答案|答案解析|解析|解答|答案是|答案为|Answer|Explanation)\s*[:：]?\s*",
        ]

        # 结束信号：出现下一题或新的题型小节标题，结束剥离
        boundary_patterns = [
            r"^\s*(?:题目|选择题|判断题|简答题)\s*\d+",
            r"^\s*(?:选择题|判断题|简答题)\s*[：:]\s*$",
            r"^\s*\d+\s*[、\.\)．]",  # 1.  1)  1．  1、
        ]

        start_regexes = [re.compile(p, re.IGNORECASE) for p in start_patterns]
        inline_regexes = [re.compile(p, re.IGNORECASE) for p in inline_patterns]
        boundary_regexes = [re.compile(p) for p in boundary_patterns]

        in_strip_block = False

        def is_start(line: str) -> bool:
            return any(r.match(line) for r in start_regexes) or any(r.search(line) for r in inline_regexes)

        def is_boundary(line: str) -> bool:
            return any(r.match(line) for r in boundary_regexes)

        for line in lines:
            if in_strip_block:
                # 检测是否到达下一题/小节
                if is_boundary(line):
                    in_strip_block = False
                    filtered.append(line)
                else:
                    # 仍在“解析/答案”块内，整行丢弃
                    continue
            else:
                if is_start(line):
                    # 进入剥离块，不输出该行
                    in_strip_block = True
                    continue
                filtered.append(line)

        # 去除末尾多余空行
        result = "\n".join(filtered)
        result = re.sub(r"\n{3,}", "\n\n", result).strip("\n")
        return result


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
                user_input = input("\n请输入您的出题需求: ").strip()
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("\n👋 谢谢使用，再见！")
                    break
                if not user_input:
                    print("⚠️  请输入有效的出题需求")
                    continue
                
                result = agent.process_request(user_input)
                
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


if __name__ == "__main__":
    main() 