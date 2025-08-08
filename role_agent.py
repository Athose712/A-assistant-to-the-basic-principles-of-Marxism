import os
from typing import Optional
from flask import Flask, request, jsonify, render_template
import uuid

# Import the new base class
from common_utils.base_dialogue_agent import BaseDialogueAgent, DialogueGraphState
from common_utils.multimodal_agent import SocratesMultimodalAgent

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
        
        # 初始化多模态Agent用于图片分析
        try:
            self.multimodal_agent = SocratesMultimodalAgent()
            print("[苏格拉底Agent] 多模态功能初始化成功")
        except Exception as e:
            print(f"[苏格拉底Agent] 多模态功能初始化失败: {e}")
            self.multimodal_agent = None
    
    def process_multimodal_dialogue(
        self, 
        user_input: str, 
        current_state: Optional[dict] = None,
        image_path: Optional[str] = None
    ) -> dict:
        """
        处理多模态对话（支持图片+文本输入）
        
        Args:
            user_input: 用户的文本输入
            current_state: 当前对话状态
            image_path: 图片文件路径（可选）
            
        Returns:
            对话处理结果
        """
        # 如果没有图片或多模态Agent不可用，使用普通对话处理
        if not image_path or not self.multimodal_agent:
            return self.process_dialogue(user_input, current_state)
        
        try:
            # 更新多模态Agent的对话上下文
            if current_state:
                character = current_state.get("simulated_character", "马克思")
                topic = current_state.get("current_topic", "马克思主义理论")
                self.multimodal_agent.update_dialogue_context(character, topic)
            
            # 使用多模态Agent处理图片+文本
            response = self.multimodal_agent.process_multimodal_request(user_input, image_path)
            
            # 如果是新对话，需要初始化状态
            if not current_state:
                # 从响应中推断角色和主题（简化处理）
                character = "马克思"  # 默认角色
                topic = "马克思主义理论"  # 默认主题
                
                new_state = {
                    "simulated_character": character,
                    "current_topic": topic,
                    "turn_count": 1,
                    "conversation_history": [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response}
                    ],
                    "last_image_path": image_path if image_path else None  # 添加图像上下文
                }
                
                return {
                    "status": "success",
                    "response": response,
                    "state": new_state
                }
            else:
                # 更新现有状态
                current_state["turn_count"] = current_state.get("turn_count", 0) + 1
                current_state["conversation_history"] = current_state.get("conversation_history", [])
                current_state["conversation_history"].extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response}
                ])
                
                current_state["last_image_path"] = image_path if image_path else current_state.get("last_image_path")
                
                return {
                    "status": "success",
                    "response": response,
                    "state": current_state
                }
                
        except Exception as e:
            print(f"[苏格拉底Agent] 多模态处理失败，回退到文本模式: {e}")
            return self.process_dialogue(user_input, current_state)

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
