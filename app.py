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
import re
import tempfile
from collections import Counter
import networkx as nx

# ===== 尝试导入 pyvis，如果失败则后续使用 matplotlib =====
try:
    from pyvis.network import Network
    pyvis_available = True
except ImportError:
    pyvis_available = False
    import matplotlib.pyplot as plt

# ===== 页面配置 =====
st.set_page_config(
    page_title="AI 教育平台",
    page_icon="🎓",
    layout="wide"
)

# ===== 移动端适配 CSS =====
st.markdown("""
<style>
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stChatMessage {
            margin-bottom: 0.5rem;
        }
        .stTextArea textarea {
            font-size: 16px;
        }
    }
    .voice-btn {
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ===== 初始化会话状态 =====
if "vectorstores" not in st.session_state:
    st.session_state.vectorstores = {}          # 教材名 -> vectorstore
if "current_textbook" not in st.session_state:
    st.session_state.current_textbook = None
if "current_textbook_content" not in st.session_state:
    st.session_state.current_textbook_content = ""
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []            # 用于学情分析
if "messages" not in st.session_state:
    st.session_state.messages = []              # 聊天历史

# ===== 侧边栏导航 =====
st.sidebar.title("🎓 AI 教育平台")
page = st.sidebar.radio(
    "选择功能",
    ["📚 智能助教", "📝 作文批改", "✍️ 习题生成", "📊 学情分析", "🧠 知识图谱"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ 全局设置")

# API Key 输入
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

    # ---- 教材管理区域 ----
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("上传新教材（.txt）", type=['txt'], key="upload")
    with col2:
        st.write("")  # 垂直占位
        st.write("")
        if st.button("📖 使用当前教材"):
            pass  # 下拉框会处理

    # 已有教材选择
    if st.session_state.vectorstores:
        selected = st.selectbox(
            "选择当前教材",
            list(st.session_state.vectorstores.keys()),
            index=0
        )
        if selected != st.session_state.current_textbook:
            st.session_state.current_textbook = selected
            st.rerun()

    # 处理新上传教材
    if uploaded_file is not None:
        # 读取文件内容
        try:
            textbook = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            try:
                textbook = uploaded_file.getvalue().decode("gbk")
            except UnicodeDecodeError:
                st.error("❌ 文件编码错误：请确保上传的文件是 UTF-8 或 GBK 编码的纯文本文件（.txt）")
                st.stop()

        textbook_name = uploaded_file.name
        if textbook_name not in st.session_state.vectorstores:
            with st.spinner(f"正在处理教材《{textbook_name}》..."):
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
                    persist_directory=f"./chroma_db_{textbook_name}"
                )
                st.session_state.vectorstores[textbook_name] = vectorstore
                st.session_state.current_textbook = textbook_name
                st.session_state.current_textbook_content = textbook
                st.success(f"✅ 教材《{textbook_name}》已添加")
                st.rerun()
        else:
            st.info(f"教材《{textbook_name}》已存在")
            st.session_state.current_textbook = textbook_name
            st.session_state.current_textbook_content = textbook
            st.rerun()

    # 如果没有选择教材，提示
    if not st.session_state.current_textbook:
        st.info("请先上传或选择一本教材。")
        st.stop()

    # ---- 对话界面 ----
    vectorstore = st.session_state.vectorstores[st.session_state.current_textbook]
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
    prompt_template = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ---- 自定义输入区（不含语音） ----
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_area("输入你的问题", key="chat_input", height=100, label_visibility="collapsed")
    with col2:
        st.write("")
        st.write("")
        send_btn = st.button("📤 发送", type="primary")

    # 处理发送
    if send_btn and user_input:
        # 保存到历史
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.qa_history.append({
            "question": user_input,
            "answer": None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                docs = retriever.invoke(user_input)
                answer = rag_chain.invoke(user_input)
                st.markdown(answer)
                with st.expander("📖 参考教材"):
                    for i, doc in enumerate(docs):
                        st.text(f"{i+1}. {doc.page_content[:100]}...")

        # 更新历史中的答案
        st.session_state.qa_history[-1]["answer"] = answer
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# ===== 功能2: 作文批改 =====
elif page == "📝 作文批改":
    st.title("📝 作文批改助手")
    st.markdown("---")

    if not api_key:
        st.error("请先在侧边栏输入智谱AI API Key")
        st.stop()

    col1, col2 = st.columns([1, 1])
    with col1:
        topic = st.text_input("✏️ 作文题目", placeholder="例如：我的梦想")
        grade_level = st.selectbox("📊 年级", ["小学", "初中", "高中", "大学"])
    with col2:
        word_count = st.number_input("📏 字数要求", min_value=100, max_value=1000, value=500, step=50)
        style = st.selectbox("📝 文体", ["记叙文", "议论文", "说明文", "应用文"])

    essay = st.text_area("📄 学生作文", height=300, placeholder="在这里粘贴学生的作文...")

    if st.button("✨ 开始批改", type="primary"):
        if not essay or not topic:
            st.error("请填写作文题目和内容")
        else:
            with st.spinner("AI 正在批改中..."):
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
                st.success("✅ 批改完成！")
                result_text = response.content

                # 简单提取总分显示
                score_match = re.search(r'【总分】(\d+)', result_text)
                if score_match:
                    total_score = int(score_match.group(1))
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("总分", f"{total_score}/100")
                    with c2:
                        st.metric("内容", re.search(r'【内容评分】(\d+)', result_text).group(1) if re.search(r'【内容评分】(\d+)', result_text) else "?")
                    with c3:
                        st.metric("结构", re.search(r'【结构评分】(\d+)', result_text).group(1) if re.search(r'【结构评分】(\d+)', result_text) else "?")
                    with c4:
                        st.metric("语言", re.search(r'【语言评分】(\d+)', result_text).group(1) if re.search(r'【语言评分】(\d+)', result_text) else "?")

                with st.expander("📋 详细批改结果", expanded=True):
                    st.markdown(result_text)

# ===== 功能3: 习题生成 =====
elif page == "✍️ 习题生成":
    st.title("✍️ 智能习题生成")
    st.markdown("---")

    if not api_key:
        st.error("请先在侧边栏输入智谱AI API Key")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        subject = st.selectbox("📚 科目", ["Python编程", "数学", "英语", "语文", "物理"])
        topic = st.text_input("🎯 知识点", placeholder="例如：for循环、一元二次方程、一般现在时")
    with col2:
        difficulty = st.select_slider("📊 难度", options=["入门", "简单", "中等", "困难", "挑战"])
        question_type = st.multiselect("📝 题型", ["选择题", "填空题", "简答题", "编程题"], default=["选择题"])

    count = st.number_input("📋 题目数量", min_value=1, max_value=10, value=3)

    if st.button("✨ 生成习题", type="primary"):
        if not topic:
            st.error("请输入知识点")
        else:
            with st.spinner("AI 正在出题..."):
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
                st.success("✅ 习题生成完成！")
                exercises = response.content.split("---")
                for i, ex in enumerate(exercises):
                    if ex.strip():
                        with st.expander(f"📌 第{i+1}题", expanded=i==0):
                            st.markdown(ex)

# ===== 功能4: 学情分析 =====
elif page == "📊 学情分析":
    st.title("📊 学情分析")
    st.markdown("---")

    if len(st.session_state.qa_history) == 0:
        st.info("暂无问答记录，请先在智能助教中提问。")
    else:
        st.subheader(f"总提问数：{len(st.session_state.qa_history)}")

        # 关键词统计
        try:
            import jieba
            all_questions = " ".join([item["question"] for item in st.session_state.qa_history])
            words = jieba.lcut(all_questions)
            stopwords = set(["的", "了", "是", "在", "和", "有", "这个", "那个", "什么", "怎么", "如何", "为什么", "吗", "呢", "吧", "啊"])
            keywords = [w for w in words if len(w) > 1 and w not in stopwords]
            counter = Counter(keywords).most_common(10)

            st.subheader("🔍 高频关键词")
            for word, count in counter:
                st.write(f"{word} : {count}次")
        except ImportError:
            st.warning("未安装 jieba 分词库，无法进行关键词分析。")

        # 最近问答
        st.subheader("📜 最近问答")
        for qa in st.session_state.qa_history[-10:]:
            with st.expander(f"Q: {qa['question'][:50]}..."):
                st.write(f"**时间**：{qa['timestamp']}")
                st.write(f"**A**: {qa['answer']}")

# ===== 功能5: 知识图谱 =====
elif page == "🧠 知识图谱":
    st.title("🧠 知识点图谱")
    st.markdown("---")

    if not st.session_state.current_textbook_content:
        st.info("请先在智能助教中上传一本教材。")
        st.stop()

    # 提取章节标题（简单正则）
    text = st.session_state.current_textbook_content
    chapters = re.findall(r'第[一二三四五六七八九十\d]+章\s*([^\n]+)', text)
    sections = re.findall(r'\d+\.\d+\s+([^\n]+)', text)
    nodes = chapters + sections

    if len(nodes) == 0:
        st.warning("未能从教材中提取出章节标题，请检查教材格式。")
    else:
        st.subheader(f"共提取到 {len(nodes)} 个知识点")

        # 构建简单图（章节之间顺序连接）
        G = nx.DiGraph()
        for i, node in enumerate(nodes):
            G.add_node(node, label=node, size=20)
            if i > 0:
                G.add_edge(nodes[i-1], node)

        if pyvis_available:
            # 使用 pyvis 生成交互式 HTML
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
            net.from_nx(G)
            net.toggle_physics(False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
        else:
            # 使用 matplotlib 生成静态图
            st.warning("⚠️ 未安装 pyvis 库，将使用静态图显示。如需交互式图谱，请运行 `pip install pyvis` 并重启应用。")
            plt.figure(figsize=(10, 6))
            pos = nx.spring_layout(G, k=1, iterations=50)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray',
                    node_size=1500, font_size=8, arrows=True)
            plt.title("知识点图谱（静态）")
            st.pyplot(plt)