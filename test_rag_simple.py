from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

print("🚀 开始测试...")

# 1. 准备一小段示例文本
sample_text = """
Python是一种解释型、面向对象的高级编程语言。
Python由Guido van Rossum于1989年发明。
Python语法简洁清晰，强制用缩进。
Python广泛应用于Web开发、数据分析、人工智能等领域。
"""

# 2. 分割文本
print("📖 分割文本...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
texts = text_splitter.split_text(sample_text)
print(f"分割成 {len(texts)} 段")

# 3. 创建 embeddings
print("🔧 加载 embeddings 模型...")
start = time.time()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print(f"✅ embeddings 加载完成，耗时 {time.time()-start:.2f} 秒")

# 4. 创建向量数据库
print("🗄️ 创建向量数据库...")
start = time.time()
vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings)
print(f"✅ 向量数据库创建完成，耗时 {time.time()-start:.2f} 秒")

# 5. 检索测试
print("🔍 检索测试...")
retriever = vectorstore.as_retriever()
docs = retriever.get_relevant_documents("Python是什么？")

print("\n📝 检索结果：")
for i, doc in enumerate(docs):
    print(f"{i+1}. {doc.page_content}")

print("\n✅ 测试完成！")