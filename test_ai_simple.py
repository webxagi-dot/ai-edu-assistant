from langchain_community.chat_models import ChatZhipuAI

# 你的API密钥
api_key = "1d9ee499e7bb413aaabe015a87b7773c.3UrwmR1C6Ew1gfDy"

# 初始化AI
llm = ChatZhipuAI(
    api_key=api_key,
    model="glm-4-flash"
)

# 直接提问
print("正在调用AI...")
response = llm.invoke("用一句话介绍Python")
print("AI回答：", response.content)