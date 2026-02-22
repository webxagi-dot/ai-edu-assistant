import streamlit as st
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# ===== 页面配置 =====
st.set_page_config(
    page_title="AI 教育助教",
    page_icon="🎓",
    layout="wide"
)

# ===== 标题 =====
st.title("🎓 AI 教育助教")
st.markdown("---")

# ===== 侧边栏 =====
with st.sidebar:
    st.header("📚 教材管理")
    
    # 教材上传
    uploaded_file = st.file_uploader(
        "上传教材文件（.txt）",
        type=['txt'],
        help="上传你的教材文本文件"
    )
    
    st.markdown("---")
    st.header("⚙️ 设置")
    
    # API Key 输入
    api_key = st.text_input(
        "智谱AI API Key",
        type="password",
        value="1d9ee499e7bb413aaabe015a87b7773c.3UrwmR1C6Ew1gfDy",
        help="输入你的智谱AI API密钥"
    )
    
    # 温度调节
    temperature = st.slider(
        "回答温度",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="值越低回答越准确，值越高越有创意"
    )
    
    st.markdown("---")
    st.markdown("### 📖 关于")
    st.info(
        "这是一个基于RAG技术的AI教育助教。\n\n"
        "它会基于你上传的教材内容回答问题，"
        "每次回答都会显示参考来源。"
    )

# ===== 主界面 =====
if uploaded_file is not None:
    # 读取上传的教材
    textbook = uploaded_file.getvalue().decode("utf-8")
    st.success(f"✅ 已加载教材：{uploaded_file.name}")
    
    # 显示教材预览
    with st.expander("📖 教材预览"):
        st.text(textbook[:500] + "..." if len(textbook) > 500 else textbook)
    
    # ===== 初始化 RAG 系统 =====
    with st.spinner("🔧 正在初始化 AI 助教..."):
        # 设置离线环境变量
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        # 本地模型路径
        local_model_path = os.path.expanduser("~/Desktop/ai-edu/local_models/all-MiniLM-L6-v2")
        
        # 分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；"]
        )
        texts = text_splitter.split_text(textbook)
        
        # 加载 embeddings（使用本地路径）
        embeddings = HuggingFaceEmbeddings(
            model_name=local_model_path,
            model_kwargs={'device': 'cpu'}
        )
        
        # 创建向量数据库
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            persist_directory="./chroma_db_web"
        )
        
        # 创建检索器
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 初始化 AI
        llm = ChatZhipuAI(
            api_key=api_key,
            model="glm-4-flash",
            temperature=temperature
        )
        
        # 提示词模板
        template = """你是一个耐心的老师。请基于以下教材内容回答学生的问题。

教材内容：
{context}

学生问题：{question}

要求：
1. 如果教材中有相关内容，请基于教材准确回答
2. 如果教材中没有，请说"教材中没有找到，不过根据我的理解："
3. 用简单易懂的语言，可以举例说明

你的回答："""
        
        prompt = PromptTemplate.from_template(template)
        
        # 格式化函数
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        # RAG 链
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        st.success("✅ AI 助教准备就绪！")
    
    # ===== 对话界面 =====
    st.markdown("---")
    st.header("💬 开始提问")
    
    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📖 参考教材"):
                    for i, source in enumerate(message["sources"]):
                        st.text(f"{i+1}. {source[:100]}...")
    
    # 输入框
    if prompt := st.chat_input("输入你的问题..."):
        # 显示用户问题
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # AI 回答
        with st.chat_message("assistant"):
            with st.spinner("🤔 思考中..."):
                # 检索相关文档
                docs = retriever.invoke(prompt)
                
                # 生成回答
                answer = rag_chain.invoke(prompt)
                
                st.markdown(answer)
                
                # 显示参考来源
                sources = [doc.page_content for doc in docs]
                with st.expander("📖 参考教材"):
                    for i, source in enumerate(sources):
                        st.text(f"{i+1}. {source[:100]}...")
        
        # 保存到历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

else:
    # 未上传教材时的提示
    st.info("👈 请在左侧边栏上传你的教材文件（.txt）开始使用")
    
    # 显示示例
    st.markdown("---")
    st.markdown("### 📝 示例教材格式")
    example = """
第1章 Python基础

1.1 变量和数据类型
变量是存储数据的容器。Python是动态类型语言，不需要声明类型。
常用数据类型：整数(int)、浮点数(float)、字符串(str)、布尔值(bool)、列表(list)、字典(dict)。

1.2 条件判断
if语句用于条件判断：
if 条件:
    执行代码
else:
    执行其他代码
    """
    st.code(example, language="text")