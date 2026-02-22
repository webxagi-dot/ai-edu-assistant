from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# ===== 1. 首先设置所有离线环境变量 =====
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ===== 2. 找到本地缓存的真实路径 =====
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
print(f"📁 本地缓存目录: {cache_dir}")

# 查找实际下载的模型快照路径
import glob
model_paths = glob.glob(f"{cache_dir}models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*/", recursive=True)
if model_paths:
    local_model_path = model_paths[0]
    print(f"✅ 找到本地模型: {local_model_path}")
else:
    local_model_path = None
    print("⚠️ 未找到本地模型缓存，将尝试从缓存加载")

# ===== 3. 配置 API 密钥 =====
ZHIPU_API_KEY = "1d9ee499e7bb413aaabe015a87b7773c.3UrwmR1C6Ew1gfDy"

# ===== 4. 读取教材 =====
print("📖 正在加载教材...")
with open("textbook.txt", "r", encoding="utf-8") as f:
    textbook = f.read()

# ===== 5. 分割文本 =====
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "；"]
)
texts = text_splitter.split_text(textbook)
print(f"✅ 分割成 {len(texts)} 个知识片段")

# ===== 6. 加载 embeddings（使用本地路径）=====
print("🔧 加载 embeddings 模型（从本地缓存）...")

# 如果找到了本地路径，直接使用
if local_model_path:
    embeddings = HuggingFaceEmbeddings(
        model_name=local_model_path,  # 直接使用本地绝对路径！
        model_kwargs={'device': 'cpu'},
        cache_folder=cache_dir
    )
else:
    # 否则回退到标准方式
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        cache_folder=cache_dir
    )

# ===== 7. 向量数据库 =====
persist_dir = "./chroma_db"

if os.path.exists(persist_dir):
    print("🗄️ 加载已有知识库...")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    print("✅ 加载完成")
else:
    print("🗄️ 创建新知识库...")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    print("✅ 创建完成")

# ===== 8. 创建检索器 =====
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ===== 9. 初始化 AI =====
llm = ChatZhipuAI(
    api_key=ZHIPU_API_KEY,
    model="glm-4-flash",
    temperature=0.3
)

# ===== 10. 提示词模板 =====
template = """你是一个耐心的Python老师。请基于以下教材内容回答学生的问题。

教材内容：
{context}

学生问题：{question}

要求：
1. 如果教材中有相关内容，请基于教材准确回答
2. 如果教材中没有，请说"教材中没有找到，不过根据我的理解："
3. 用简单易懂的语言，可以举例说明

你的回答："""

prompt = PromptTemplate.from_template(template)

# ===== 11. 格式化函数 =====
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# ===== 12. RAG 链 =====
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ===== 13. 开始对话 =====
print("\n" + "="*50)
print("🎓 AI助教（基于教材版 - 绝对离线）")
print("="*50)
print("✅ 已通过本地路径加载模型")
print("我会基于 textbook.txt 回答你的问题")
print("输入 'quit' 退出")
print("-"*50)

while True:
    question = input("\n👨‍🎓 学生: ")
    
    if question.lower() == 'quit':
        print("👋 再见！")
        break
    
    print("🤖 正在检索教材...")
    
    # 检索相关文档
    docs = retriever.invoke(question)
    
    # 调用 RAG 链
    answer = rag_chain.invoke(question)
    
    print(f"\n💡 老师: {answer}")
    
    # 显示参考来源
    print("\n📖 参考教材段落：")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.page_content[:50]}...")
    print("-"*50)