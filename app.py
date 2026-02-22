import streamlit as st
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import time

# ===== 页面配置 =====
st.set_page_config(
    page_title="AI 教育平台",
    page_icon="🎓",
    layout="wide"
)

# ===== 侧边栏导航 =====
st.sidebar.title("🎓 AI 教育平台")
page = st.sidebar.radio(
    "选择功能",
    ["📚 智能助教", "📝 作文批改", "✍️ 习题生成"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ 全局设置")

# API Key 输入（从环境变量读取默认值）
default_key = os.getenv("ZHIPU_API_KEY", "1d9ee499e7bb413aaabe015a87b7773c.3UrwmR1C6Ew1gfDy")
api_key = st.sidebar.text_input(
    "智谱AI API Key",
    type="password",
    value=default_key
)

# 温度调节
temperature = st.sidebar.slider(
    "回答温度",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.1
)

# ===== 初始化 AI =====
@st.cache_resource
def get_llm():
    return ChatZhipuAI(
        api_key=api_key,
        model="glm-4-flash",
        temperature=temperature
    )

if api_key:
    llm = get_llm()

# ===== 功能1: 智能助教 =====
if page == "📚 智能助教":
    st.title("📚 智能助教")
    st.markdown("---")
    
    # 教材上传
    uploaded_file = st.file_uploader(
        "上传教材文件（.txt）",
        type=['txt'],
        key="textbook_uploader"
    )
    
    if uploaded_file is not None:
        # 处理编码
        try:
            textbook = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            try:
                textbook = uploaded_file.getvalue().decode("gbk")
            except UnicodeDecodeError:
                st.error("❌ 文件编码错误：请确保上传的文件是 UTF-8 或 GBK 编码的纯文本文件（.txt）")
                st.stop()
        
        st.success(f"✅ 已加载教材：{uploaded_file.name}")
        
        with st.expander("📖 教材预览"):
            st.text(textbook[:500] + "..." if len(textbook) > 500 else textbook)
        
        # 初始化 RAG
        with st.spinner("🔧 初始化知识库..."):
            # 分割文本
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=50,
                separators=["\n\n", "\n", "。", "；"]
            )
            texts = text_splitter.split_text(textbook)
            
            # 加载 embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # 创建向量数据库
            vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                persist_directory="./chroma_db_assistant"
            )
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
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
            
            def format_docs(docs):
                return "\n\n".join([doc.page_content for doc in docs])
            
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            st.success("✅ 知识库准备就绪！")
        
        # 对话界面
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("输入你的问题..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    docs = retriever.invoke(prompt)
                    answer = rag_chain.invoke(prompt)
                    st.markdown(answer)
                    
                    with st.expander("📖 参考教材"):
                        for i, doc in enumerate(docs):
                            st.text(f"{i+1}. {doc.page_content[:100]}...")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

# ===== 功能2: 作文批改 =====
elif page == "📝 作文批改":
    st.title("📝 作文批改助手")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        topic = st.text_input("✏️ 作文题目", placeholder="例如：我的梦想")
        grade_level = st.selectbox(
            "📊 年级",
            ["小学", "初中", "高中", "大学"]
        )
        
    with col2:
        word_count = st.number_input("📏 字数要求", min_value=100, max_value=1000, value=500, step=50)
        style = st.selectbox(
            "📝 文体",
            ["记叙文", "议论文", "说明文", "应用文"]
        )
    
    essay = st.text_area(
        "📄 学生作文",
        height=300,
        placeholder="在这里粘贴学生的作文..."
    )
    
    if st.button("✨ 开始批改", type="primary"):
        if not essay or not topic:
            st.error("请填写作文题目和内容")
        else:
            with st.spinner("AI 正在批改中..."):
                # 构建批改提示词
                grading_prompt = f"""你是一位经验丰富的语文老师。请对以下作文进行批改。

作文题目：{topic}
年级：{grade_level}
字数要求：{word_count}字
文体：{style}

学生作文：
{essay}

请从以下几个方面进行批改，并给出百分制总分：

1. 内容（30分）：主题明确、内容充实、观点清晰
2. 结构（30分）：层次分明、过渡自然、逻辑清晰
3. 语言（30分）：表达准确、词汇丰富、句式多样
4. 创意（10分）：有新意、有特色

请按以下格式输出：
【总分】XX分
【内容评分】XX分 评语：...
【结构评分】XX分 评语：...
【语言评分】XX分 评语：...
【创意评分】XX分 评语：...
【详细评语】...
【修改建议】...
【示范段落】...
"""
                
                response = llm.invoke(grading_prompt)
                
                # 显示结果
                st.success("✅ 批改完成！")
                
                # 解析并显示评分
                result_text = response.content
                
                # 尝试提取总分
                import re
                score_match = re.search(r'【总分】(\d+)', result_text)
                if score_match:
                    total_score = int(score_match.group(1))
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("总分", f"{total_score}/100")
                    with col2:
                        st.metric("内容", re.search(r'【内容评分】(\d+)', result_text).group(1) if re.search(r'【内容评分】(\d+)', result_text) else "?")
                    with col3:
                        st.metric("结构", re.search(r'【结构评分】(\d+)', result_text).group(1) if re.search(r'【结构评分】(\d+)', result_text) else "?")
                    with col4:
                        st.metric("语言", re.search(r'【语言评分】(\d+)', result_text).group(1) if re.search(r'【语言评分】(\d+)', result_text) else "?")
                
                # 显示完整批改结果
                with st.expander("📋 详细批改结果", expanded=True):
                    st.markdown(result_text)

# ===== 功能3: 习题生成 =====
elif page == "✍️ 习题生成":
    st.title("✍️ 智能习题生成")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        subject = st.selectbox(
            "📚 科目",
            ["Python编程", "数学", "英语", "语文", "物理"]
        )
        
        topic = st.text_input("🎯 知识点", placeholder="例如：for循环、一元二次方程、一般现在时")
        
    with col2:
        difficulty = st.select_slider(
            "📊 难度",
            options=["入门", "简单", "中等", "困难", "挑战"]
        )
        
        question_type = st.multiselect(
            "📝 题型",
            ["选择题", "填空题", "简答题", "编程题"],
            default=["选择题"]
        )
    
    count = st.number_input("📋 题目数量", min_value=1, max_value=10, value=3)
    
    if st.button("✨ 生成习题", type="primary"):
        if not topic:
            st.error("请输入知识点")
        else:
            with st.spinner("AI 正在出题..."):
                # 构建出题提示词
                exercise_prompt = f"""你是一位经验丰富的{subject}老师。请根据以下要求生成练习题。

科目：{subject}
知识点：{topic}
难度：{difficulty}
题型：{', '.join(question_type)}
题目数量：{count}

要求：
1. 题目要覆盖知识点的核心内容
2. 难度要适合{difficulty}水平
3. 题型要多样化
4. 提供参考答案和解析
5. 题目表述要清晰准确

请按以下格式输出每个题目：
【题目X】
题型：[题型]
题目：[题目内容]
答案：[参考答案]
解析：[详细解析]
---
"""
                
                response = llm.invoke(exercise_prompt)
                
                # 显示结果
                st.success("✅ 习题生成完成！")
                
                # 分割并显示题目
                exercises = response.content.split("---")
                for i, exercise in enumerate(exercises):
                    if exercise.strip():
                        with st.expander(f"📌 第{i+1}题", expanded=i==0):
                            st.markdown(exercise)
                            
                            if st.button(f"查看答案", key=f"ans_{i}"):
                                st.info("答案已在题目中显示")