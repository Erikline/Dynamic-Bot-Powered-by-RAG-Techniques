import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import re
import tempfile
import streamlit as st
from PIL import Image
import logging
import warnings
from dotenv import load_dotenv
import torch
import shutil
import hashlib


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

# --- 配置 ---
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"

# --- 模型配置 ---
MODEL_ID = "THUDM/GLM-4-9B-0414"

# --- 持久化配置 ---
# 获取当前脚本所在的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data", "chroma_db")
PERSIST_STATE_FILE = os.path.join(BASE_DIR, "data", "processed_files_state.json")

# --- 应用设置 ---
APP_TITLE = f"使用 Deepseek-R1 和 基于 BGE 词嵌入 的 ChromaDB 的 RAG 应用 (支持持久化与文件同步)"
FAVICON_PATH = "Bot.png"
LOGO_PATH = "icon.png"

# --- Streamlit 页面配置 ---
try:
    if os.path.exists(FAVICON_PATH):
        favicon = Image.open(FAVICON_PATH)
        st.set_page_config(page_title=APP_TITLE, page_icon=favicon, layout="wide")
    else:
        st.set_page_config(page_title=APP_TITLE, layout="wide")
        # 仅在第一次运行时警告，避免刷新时一直弹窗
        # st.warning(f"在相对路径 '{FAVICON_PATH}' 未找到 Favicon 图标。")
except Exception as e:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    logging.warning(f"无法加载 Favicon: {e}")

# --- Streamlit 侧边栏 ---
try:
    if os.path.exists(LOGO_PATH):
        st.sidebar.image(LOGO_PATH, use_container_width=True)
    else:
        # st.sidebar.warning(f"未找到 Logo 图片: {LOGO_PATH}")
        pass
except Exception as e:
     st.sidebar.warning(f"无法加载 Logo 图片: {e}")

with st.sidebar:
    st.markdown(f"**{APP_TITLE}**")
    
    st.divider()

    # 添加一个按钮用于清除缓存和持久化数据
    if st.button("清除所有数据缓存 (包括 ChromaDB)", key="clear_cache_and_db"):
         st.cache_resource.clear() # 清除 Streamlit 的资源缓存
         if os.path.exists(PERSIST_DIRECTORY) or os.path.exists(PERSIST_STATE_FILE):
             st.info(f"正在清理旧的数据目录...")
             try:
                 if os.path.exists(PERSIST_DIRECTORY):
                    shutil.rmtree(PERSIST_DIRECTORY) 
                 
                 # 同时也删除状态记录文件，确保逻辑重置
                 if os.path.exists(PERSIST_STATE_FILE):
                     os.remove(PERSIST_STATE_FILE)

                 st.sidebar.success("已清除缓存和持久化数据。请刷新页面。")
                 st.rerun() 
             except Exception as e:
                 st.sidebar.error(f"清除目录失败: {e}")
                 logging.error(f"清除失败: {e}", exc_info=True)
         else:
             st.sidebar.info("没有找到需要清除的持久化数据。")

# --- 文件处理辅助函数 ---
def calculate_file_hash(uploaded_file):
    """计算上传文件的 SHA256 哈希值。"""
    sha256 = hashlib.sha256()
    uploaded_file.seek(0)
    while True:
        data = uploaded_file.read(8192)
        if not data:
            break
        sha256.update(data)
    uploaded_file.seek(0)
    return sha256.hexdigest()

def load_processed_state():
    """从状态文件加载之前处理的文件状态（文件名和哈希值）。"""
    if os.path.exists(PERSIST_STATE_FILE):
        try:
            with open(PERSIST_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"加载文件状态文件失败 ({PERSIST_STATE_FILE}): {e}")
            return {}
    return {}

def save_processed_state(state_dict):
    """将当前文件状态（文件名和哈希值）保存到状态文件。"""
    # 确保父目录存在
    os.makedirs(os.path.dirname(PERSIST_STATE_FILE), exist_ok=True)
    try:
        with open(PERSIST_STATE_FILE, 'w') as f:
            json.dump(state_dict, f, indent=4)
    except Exception as e:
        logging.error(f"保存文件状态文件失败 ({PERSIST_STATE_FILE}): {e}", exc_info=True)


# --- 核心 RAG 功能 (configure_retriever) ---
@st.cache_resource(ttl="2h") 
def configure_retriever(uploaded_files):
    """
    基于上传的文件或现有的持久化数据配置并返回一个 ChromaDB 检索器。
    返回一个元组：(检索器对象, 是否从磁盘加载的布尔值)
    """
    temp_dir = None
    chroma_retriever = None
    was_loaded_from_disk = False

    # 1. 创建 Embedding 模型
    try:
        st.info("正在初始化 BGE 嵌入模型 (用于 Chroma)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"检测到可用设备: {device}")
        
        # 新版 langchain-huggingface 写法
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        st.success("BGE 嵌入模型初始化成功。")

    except Exception as e:
        st.error(f"初始化 BGE 嵌入模型失败: {e}")
        logging.error("Embedding 初始化失败:", exc_info=True)
        return None, False

    # 2. 检查文件状态并决定加载或重建
    current_files_state = {}
    if uploaded_files:
        st.info("正在计算上传文件的哈希值...")
        try:
            for file in uploaded_files:
                current_files_state[file.name] = calculate_file_hash(file)
        except Exception as e:
            st.error(f"计算文件哈希值失败: {e}")
            return None, False

    previous_files_state = load_processed_state()

    needs_rebuild = False
    if uploaded_files:
        if not previous_files_state:
            st.info("未找到之前的文档数据状态，将创建新的向量存储。")
            needs_rebuild = True
        elif current_files_state != previous_files_state:
            st.info("检测到上传文件与之前保存的数据状态不一致，将重新创建向量存储。")
            needs_rebuild = True
        else:
            st.info("上传文件与之前保存的数据状态一致，将尝试加载现有的向量存储。")
            needs_rebuild = False
    elif previous_files_state:
        st.info("未上传新文档，尝试加载历史数据...")
        needs_rebuild = False
    else:
        st.warning("没有上传新的文档，也未找到之前的文档数据。请上传 PDF 文档以开始。")
        return None, False

    # 3. 根据判断结果执行加载或重建
    if needs_rebuild:
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                st.success("旧的 Chroma 目录已清理。")
            except Exception as cleanup_e:
                st.error(f"清理旧的 Chroma 目录失败: {cleanup_e}")

        all_splits = []
        try:
            temp_dir = tempfile.TemporaryDirectory()
            st.info(f"开始处理 {len(uploaded_files)} 个上传的文件...")
            
            # 进度条
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                temp_filepath = os.path.join(temp_dir.name, file.name)
                try:
                    file.seek(0)
                    with open(temp_filepath, "wb") as f:
                        f.write(file.getvalue())
                    
                    loader = PyPDFLoader(temp_filepath)
                    file_docs = loader.load()
                    if not file_docs:
                         logging.warning(f"文件 {file.name} 为空。")
                         continue
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    splits = text_splitter.split_documents(file_docs)
                    
                    # 为文档添加元数据源
                    for split in splits:
                        split.metadata["source"] = file.name
                        
                    all_splits.extend(splits)
                    
                    # 更新进度
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.error(f"处理 PDF {file.name} 出错: {e}")

            if not all_splits:
                st.warning("没有成功处理任何文档。")
                return None, False

            try:
                st.info(f"正在创建 Chroma 向量存储并持久化到 {PERSIST_DIRECTORY}...")
                
                # 新版 langchain-chroma 写法
                vectordb = Chroma.from_documents(
                    documents=all_splits,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIRECTORY
                )
                
                st.success(f"Chroma 向量存储创建成功。")

                save_processed_state(current_files_state)
                st.success(f"已保存当前文件状态。")

                chroma_retriever = vectordb.as_retriever(
                    search_type="mmr", 
                    search_kwargs={"k": 6} # 这里我调整了 k 值，你可以改回去
                )
                return chroma_retriever, False

            except Exception as e:
                st.error(f"创建 Chroma 向量存储失败: {e}")
                return None, False

        except Exception as e:
            st.error(f"处理过程中发生错误: {e}")
            return None, False
        finally:
            if temp_dir:
                try:
                    temp_dir.cleanup()
                except:
                    pass

    else: # 尝试从磁盘加载
        if not os.path.exists(PERSIST_DIRECTORY):
             st.warning(f"持久化目录不存在。请上传文件。")
             return None, False

        try:
            st.info(f"正在加载 Chroma 向量存储...")
            # 新版加载方式
            vectordb = Chroma(
                persist_directory=PERSIST_DIRECTORY, 
                embedding_function=embeddings
            )
            
            chroma_retriever = vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6}
            )
            st.success("Chroma 检索器加载成功。")
            was_loaded_from_disk = True
            return chroma_retriever, was_loaded_from_disk

        except Exception as e:
            st.error(f"加载失败 ({e})。可能是数据版本不兼容，请点击'清除所有数据缓存'。")
            return None, False


# --- Streamlit 应用主逻辑 ---

# 1. 文件上传
uploaded_files = st.sidebar.file_uploader(
    label="上传 PDF 文件",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader_main"
)

# 2. 配置检索器
retriever = None 
loaded_from_disk = False 

if uploaded_files or os.path.exists(PERSIST_DIRECTORY):
    with st.spinner("正在初始化检索器..."):
        retriever, loaded_from_disk = configure_retriever(uploaded_files)

# 3. 初始化 LLM
llm = None 
if retriever is not None:
    try:
        st.info(f"正在初始化 LLM: {MODEL_ID} (SiliconFlow)...")
        
        # 根据你的 curl 命令配置参数
        llm = ChatOpenAI(
            model_name=MODEL_ID,
            openai_api_key=SILICONFLOW_API_KEY,
            openai_api_base=SILICONFLOW_API_BASE,
            
            # 标准参数
            temperature=0.7,        # curl 中的设置
            max_tokens=4096,        # curl 中的设置
            
            # 额外参数 (对应 curl 中的 top_p, top_k, frequency_penalty 等)
            model_kwargs={
                "top_p": 0.7,
                # "top_k": 50,
                "frequency_penalty": 0.5,
                # "min_p": 0.05, # LangChain 部分版本可能不支持传这个，如果报错请注释掉
            }
        )
        st.success(f"LLM ({MODEL_ID}) 初始化成功。")
    except Exception as e:
        st.error(f"初始化 LLM 失败: {e}")
        st.stop()

# 4. 初始化聊天记录 (使用 LangChain 的 Streamlit 历史记录类)
msgs = StreamlitChatMessageHistory(key="chat_messages_history")

# 5. 定义 Prompt
RESPONSE_TEMPLATE = """<s>[INST]
<<SYS>>
你是一个专业、耐心、且乐于助人的 AI 助手。请根据下面提供的上下文信息来回答用户的问题。
如果上下文信息足以回答问题，请直接依据上下文进行回答。
如果上下文信息不足以回答问题，或者问题与上下文无关，请明确告知用户上下文信息不足。
回答应清晰、简洁，并使用中文。
<<SYS>>

上下文信息:
---
{context}
---

用户问题: {question}
[/INST]
AI 助手回答:
"""
PROMPT = PromptTemplate(template=RESPONSE_TEMPLATE, input_variables=["context", "question"])

# 6. 创建 RAG 问答链
qa_chain = None 
if llm is not None and retriever is not None:
    try:
        # st.info("正在创建 RAG 问答链...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"创建 RAG 链失败: {e}")
        st.stop()

# 7. 显示聊天界面

# 初始欢迎语
if len(msgs.messages) == 0:
    if retriever is not None: 
        if loaded_from_disk:
            initial_message = f"📚 已加载历史文档库。请问有什么关于这些文档的问题吗？"
        elif uploaded_files:
            initial_message = f"📄 文档已处理完毕。请问有什么问题？"
        else:
            initial_message = "请上传文档。"
    else:
        initial_message = "👋 你好！请在左侧上传 PDF 文档，我将根据文档内容回答你的问题。"
        
    msgs.add_ai_message(initial_message)

# 头像映射
avatars = {"human": "🧑‍💻", "ai": "🤖"}

# 渲染历史消息
for msg in msgs.messages:
    st.chat_message(msg.type, avatar=avatars.get(msg.type)).write(msg.content)

# 处理用户输入
chat_input_disabled = (qa_chain is None)
placeholder_text = "请先上传文档..." if chat_input_disabled else "输入你的问题..."

if user_query := st.chat_input(placeholder=placeholder_text, disabled=chat_input_disabled):
    msgs.add_user_message(user_query)
    st.chat_message("human", avatar=avatars["human"]).write(user_query)

    with st.chat_message("ai", avatar=avatars["ai"]):
        placeholder = st.empty()
        placeholder.markdown("⏳ 正在思考...")
        try:
            # 调用 QA 链
            response = qa_chain.invoke({"query": user_query})
            answer = response.get("result")
            source_docs = response.get("source_documents", [])

            if not answer:
                 answer = "抱歉，未能生成回答。"

            placeholder.markdown(answer)
            msgs.add_ai_message(answer)

            # 显示来源
            if source_docs:
                with st.expander("🔍 查看检索到的上下文来源", expanded=False):
                     for i, doc in enumerate(source_docs):
                        source = doc.metadata.get('source', '未知文件')
                        # 简化文件名显示
                        source_name = os.path.basename(source)
                        page = doc.metadata.get('page', '?')
                        if isinstance(page, int): page += 1
                        
                        st.markdown(f"**📄 来源 {i+1}:** `{source_name}` (第 {page} 页)")
                        content_preview = doc.page_content[:300].replace("\n", " ")
                        st.caption(f"{content_preview}...")
                        st.divider()

        except Exception as e:
            placeholder.empty()
            st.error(f"发生错误: {e}")
            msgs.add_ai_message(f"发生错误: {e}")

# --- 侧边栏 "关于" 部分 ---
with st.sidebar:
    st.divider()
    about = st.expander("关于此应用")
    
    status_text = "未知"
    if retriever:
        if loaded_from_disk: status_text = "已加载历史数据"
        else: status_text = "使用当前上传数据"
    else:
        status_text = "等待上传"

    about.write(f"""
    **状态:** {status_text}
    
    **技术栈:**
    *   LLM: {MODEL_ID}
    *   Embedding: BGE-Large-Zh
    *   VectorDB: Chroma ({'持久化开启' if os.path.exists(PERSIST_DIRECTORY) else '无数据'})
    
    **功能:**
    文件内容会自动持久化保存。下次打开无需重新上传，除非文件发生变动。
    """)

















