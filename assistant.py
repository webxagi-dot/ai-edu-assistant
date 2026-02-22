from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

# ===== 第1步：设置API密钥（先去智谱AI官网注册获取）=====
# 注册地址：https://open.bigmodel.cn/
# 你的API密钥填写在这里（记得加引号）
ZHIPU_API_KEY = "1d9ee499e7bb413aaabe015a87b7773c.3UrwmR1C6Ew1gfDy"

# ===== 第2步：初始化AI =====
llm = ChatZhipuAI(
    api_key=ZHIPU_API_KEY,
    model="glm-4-flash",  # 免费版
    temperature=0.7
)

# ===== 第3步：定义AI老师的性格 =====
system_prompt = SystemMessage(content="""你是一个耐心的AI老师，擅长：
1. 用简单的话解释复杂概念
2. 举生活中的例子
3. 鼓励学生思考

请用中文回答，语气要温和。""")

print("="*50)
print("🎓 AI智能助教（教育版）")
print("="*50)
print("输入你的问题，输入 'quit' 退出")
print("-"*50)

# ===== 第4步：开始对话 =====
while True:
    question = input("\n👨‍🎓 学生: ")
    
    if question.lower() == 'quit':
        print("👋 再见！")
        break
    
    # 准备消息
    messages = [
        system_prompt,
        HumanMessage(content=question)
    ]
    
    print("🤖 老师正在思考...")
    
    # 调用AI
    response = llm.invoke(messages)
    
    print(f"\n💡 老师: {response.content}")
    print("-"*50)