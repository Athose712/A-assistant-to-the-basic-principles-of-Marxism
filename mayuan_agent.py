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
    
    def process_multimodal_request(self, text_input: str, image_path: Optional[str] = None) -> str:
        """
        处理多模态请求（支持图片+文本输入）
        
        Args:
            text_input: 用户的文本输入
            image_path: 图片文件路径（可选）
            
        Returns:
            AI的回复
        """
        # 如果没有图片或多模态Agent不可用，使用普通文本处理
        if not image_path or not self.multimodal_agent:
            return self.process_request(text_input)
        
        try:
            # 使用多模态Agent处理图片+文本
            return self.multimodal_agent.process_multimodal_request(text_input, image_path)
        except Exception as e:
            print(f"[马原Agent] 多模态处理失败，回退到文本模式: {e}")
            return self.process_request(text_input)


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