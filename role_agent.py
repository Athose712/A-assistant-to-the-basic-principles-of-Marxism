import os
from typing import Optional
from flask import Flask, request, jsonify, render_template
import uuid

# Import the new base class
from common_utils.base_dialogue_agent import BaseDialogueAgent, DialogueGraphState

#  API Key Setup (与之前相同) 
# 重要：运行前必须设置环境变量
# 方法1：在终端中运行
#    export DASHSCOPE_API_KEY='your_api_key_here'      # Linux/Mac
#    $env:DASHSCOPE_API_KEY='your_api_key_here'         # Windows PowerShell
# 方法2：取消下面这行注释并填入您的API Key
os.environ["DASHSCOPE_API_KEY"] = "sk-67eb31fc296f46728913a60ad6c03e32" # 请替换为您的实际API Key并取消注释

# 让 dashscope SDK 立刻获取到正确的 key（import dashscope 已经在上面执行过）
import dashscope as _ds_internal
if os.environ.get("DASHSCOPE_API_KEY"):
    _ds_internal.api_key = os.environ["DASHSCOPE_API_KEY"]



class SocratesAgent(BaseDialogueAgent):
    """特定人物语气的苏格拉底对话 Agent - 基于 BaseDialogueAgent"""
    
    def __init__(self):
        """初始化马克思主义基本原理苏格拉底对话 Agent"""
        super().__init__(
            subject_name="马克思主义基本原理",
            vectorstore_path="database_agent_mayuan",
            default_topic="马克思主义哲学",
            default_character="马克思",
            llm_model="qwen-max",
            temperature=0.8
        )

def main():
    """主程序入口 - 提供命令行交互界面"""
    print("=" * 60)
    print("      特定人物语气的苏格拉底对话 Agent")
    print("        基于 LangGraph 和 LangChain 构建")
    print("=" * 60)
    
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
        agent = SocratesAgent()
        print("\n🎉 Agent 初始化完成！")
        
        print("\n💡 使用说明:")
        print("   这是一个特定人物语气的苏格拉底对话 Agent。")
        print("   您可以指定主题和模拟人物，例如：")
        print("   - '我想和马克思探讨一下唯物辩证法。'")
        print("   - '我们来谈谈历史唯物主义，你就像恩格斯一样提问吧。'")
        print("   - '我想深入思考一下实践的本质。'")
        print("   - 输入 'quit' 或 'exit' 退出对话。")
        print("=" * 60)
        
        current_dialogue_state: Optional[DialogueGraphState] = None
        
        while True:
            try:
                user_input = input("\n您想探讨什么 (输入 'quit' 退出): ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("\n👋 谢谢使用，再见！")
                    break
                
                if not user_input:
                    print("⚠️  请输入您的问题或想探讨的主题。")
                    continue
                
                # 每次调用都传递上一次的对话状态
                response_data = agent.process_dialogue(user_input, current_dialogue_state)
                
                print("\n" + "=" * 50)
                print(f"📖 {response_data['state']['simulated_character']}（{response_data['state']['current_topic']}）的回应:")
                print("=" * 50)
                print(response_data["response"])
                print("=" * 50)
                
                # 更新当前对话状态，用于下一轮
                current_dialogue_state = response_data["state"]

                if response_data["status"] == "error":
                    print("对话遇到错误，可能需要重新开始。")
                    current_dialogue_state = None  # 清空状态，重新开始
            
            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 处理请求时出现错误: {e}")
                print("请重试或输入新的请求")
                current_dialogue_state = None  # 清空状态，重新开始
                
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        print("请检查:")
        print("1. DASHSCOPE_API_KEY 环境变量是否设置正确")
        print("2. 向量数据库文件是否存在于 'database_agent_mayuan' 目录")
        print("3. 网络连接是否正常")

if __name__ == "__main__":
    main()
