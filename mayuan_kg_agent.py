"""
马克思主义基本原理知识图谱 Agent
"""
from common_utils.base_kg_agent import BaseKnowledgeGraphAgent


class MayuanKnowledgeGraphAgent(BaseKnowledgeGraphAgent):
    """
    专门用于“马克思主义基本原理”课程的知识图谱 Agent。
    """
    def __init__(self):
        # 调用父类构造函数，传入马原课程的特定参数
        super().__init__(
            subject_name="马克思主义基本原理",
            vectorstore_path="database_agent_mayuan"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("   马克思主义基本原理知识图谱 Agent")
    print("   基于 LangChain 构建")
    print("=" * 60)

    try:
        agent = MayuanKnowledgeGraphAgent()
        print("\n🎉 Agent 初始化完成！")
        print("\n💡 输入知识点名称，如：'唯物辩证法'，输入 'quit' 退出。")

        while True:
            try:
                user_query = input("\n请输入知识点: ").strip()
                if user_query.lower() in {"quit", "exit", "q"}:
                    print("👋 再见！")
                    break
                if not user_query:
                    print("⚠️  请输入有效的知识点名称！")
                    continue

                result = agent.build_knowledge_graph(user_query)
                print("\n" + "=" * 50)
                print("Mermaid 代码及总结输出:")
                print("=" * 50)
                print(result)
                print("=" * 50)

            except KeyboardInterrupt:
                print("\n👋 程序被中断，再见！")
                break
            except Exception as e:
                print(f"❌ 处理请求时发生错误: {e}")
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}") 