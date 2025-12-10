# 导入所需的库
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

# 加载环境变量 (确保 .env 文件存在或环境变量已设置)
# 在部署时，通常通过平台的 Secrets 功能设置环境变量，而不是依赖 .env 文件
load_dotenv()

# 忽略特定的 Streamlit 弃用警告
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*use_column_width parameter has been deprecated.*"
)

# 导入 Langchain 相关模块
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings # Embeddings 保持不变
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入 DashScope (OpenAI兼容模式) 所需的组件
try:
    from langchain_openai import ChatOpenAI # <--- 使用 LangChain 的 OpenAI 包装器
except ImportError:
    st.error("找不到 langchain-openai 库。请安装它：pip install -U langchain-openai")
    st.stop()
try:
    import openai # 检查 openai 库本身是否存在
except ImportError:
    st.error("找不到 openai 库。请安装它：pip install -U openai")
    st.stop()


# --- 配置 ---
# DashScope (百炼) API 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") # 从环境变量获取
DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if not DASHSCOPE_API_KEY:
    st.error("未找到 DashScope API 密钥。请在部署平台的 Secrets 中设置环境变量 DASHSCOPE_API_KEY。")
    st.info("如何获取API Key: https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key")
    st.stop()

# --- 模型配置 ---
MODEL_ID = "deepseek-r1" # 使用 DashScope 的模型名称

# --- 持久化配置 ---
# 使用相对于脚本的相对路径，数据将保存在项目下的 data/chroma_db_streamlit 目录中
# 重要：确保将 'data/' 目录添加到你的 .gitignore 文件中！
PERSIST_DIRECTORY = "data/chroma_db_streamlit"
PERSIST_STATE_FILE = os.path.join(PERSIST_DIRECTORY, "processed_files_state.json") # os.path.join 会正确处理路径

# --- 应用设置 ---
APP_TITLE = f"使用 Deepseek-R1 和 基于 BGE 词嵌入 的 ChromaDB 的 RAG 应用 (支持持久化与文件同步)"
# --- 修改这里：使用相对于脚本的相对路径 (因为图片在根目录) ---
FAVICON_PATH = "Bot.png"
LOGO_PATH = "icon.png"
# --- 修改结束 ---

# --- Streamlit 页面配置 ---
try:
    # 检查文件是否存在，使用相对路径
    if os.path.exists(FAVICON_PATH):
        favicon = Image.open(FAVICON_PATH)
        st.set_page_config(page_title=APP_TITLE, page_icon=favicon, layout="wide")
    else:
        st.set_page_config(page_title=APP_TITLE, layout="wide")
        # 更新警告信息中的路径
        st.warning(f"在相对路径 '{FAVICON_PATH}' 未找到 Favicon 图标。请确保文件 Bot.png 与 Bot.py 在同一目录。")
except Exception as e:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.warning(f"无法加载 Favicon 图标 ({FAVICON_PATH}): {e}")


# --- Streamlit 侧边栏 ---
try:
    # 检查文件是否存在，使用相对路径
    if os.path.exists(LOGO_PATH):
        st.sidebar.image(LOGO_PATH, use_container_width=True)
    else:
        # 更新警告信息中的路径
        st.sidebar.warning(f"在相对路径 '{LOGO_PATH}' 未找到 Logo 图片。请确保文件 icon.png 与 Bot.py 在同一目录。")
except Exception as e:
     st.sidebar.warning(f"无法加载 Logo 图片 ({LOGO_PATH}): {e}")

with st.sidebar:
    st.markdown(f"**{APP_TITLE}**")
    # 添加一个按钮用于清除缓存和持久化数据（可选，用于调试或处理不同文件）
    if st.button("清除所有数据缓存 (包括 ChromaDB)", key="clear_cache_and_db"):
         st.cache_resource.clear() # 清除 Streamlit 的资源缓存
         if os.path.exists(PERSIST_DIRECTORY):
             st.info(f"正在清理旧的 Chroma 目录: {PERSIST_DIRECTORY}")
             try:
                 shutil.rmtree(PERSIST_DIRECTORY) # 清除持久化目录
                 st.sidebar.success("已清除缓存和 ChromaDB 持久化数据。请刷新页面并重新上传文件。")
                 st.rerun() # 重新运行应用
             except Exception as e:
                 st.sidebar.error(f"清除 ChromaDB 目录失败: {e}")
                 logging.error(f"清除 ChromaDB 目录失败: {e}", exc_info=True)
         else:
             st.sidebar.info("没有找到需要清除的 ChromaDB 持久化数据。")


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
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    try:
        with open(PERSIST_STATE_FILE, 'w') as f:
            json.dump(state_dict, f, indent=4)
    except Exception as e:
        logging.error(f"保存文件状态文件失败 ({PERSIST_STATE_FILE}): {e}", exc_info=True)


# --- 核心 RAG 功能 (configure_retriever - 应用 GPU 修正 & 持久化 & 文件同步) ---
@st.cache_resource(ttl="2h") # 缓存2小时
def configure_retriever(uploaded_files):
    """
    基于上传的文件或现有的持久化数据配置并返回一个 ChromaDB 检索器。
    返回一个元组：(检索器对象, 是否从磁盘加载的布尔值)
    处理文件变更逻辑。
    """
    temp_dir = None
    chroma_retriever = None
    was_loaded_from_disk = False # 新增标志

    # 1. 创建 Embedding 模型 (加载和创建都需要)
    try:
        st.info("正在初始化 BGE 嵌入模型 (用于 Chroma)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"检测到可用设备: {device}，将用于嵌入模型加载。")
        if device == 'cuda':
             try:
                st.info(f"GPU 型号: {torch.cuda.get_device_name(0)}")
             except Exception:
                st.warning("无法获取 GPU 型号信息。")

        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        st.success("BGE 嵌入模型初始化成功。")

    except Exception as e:
        st.error(f"初始化 BGE 嵌入模型失败: {e}")
        logging.error("Embedding 初始化失败:", exc_info=True)
        if "out of memory" in str(e).lower():
            st.error("GPU 显存不足！请尝试关闭其他占用显存的程序，或在代码中将 device 改为 'cpu'。")
        elif "meta tensor" in str(e):
            st.error("加载模型时遇到 Meta Tensor 错误。请尝试清理 Hugging Face 模型缓存，然后重启应用。")
        else:
             st.error("请检查模型名称、网络连接或系统资源。")
        return None, False

    # 2. 检查文件状态并决定加载或重建
    current_files_state = {}
    if uploaded_files:
        st.info("正在计算上传文件的哈希值以检查是否有变更...")
        try:
            for file in uploaded_files:
                current_files_state[file.name] = calculate_file_hash(file)
            st.success(f"已计算 {len(uploaded_files)} 个文件的哈希值。")
        except Exception as e:
            st.error(f"计算文件哈希值失败: {e}")
            logging.error("计算哈希失败:", exc_info=True)
            st.warning("文件哈希计算失败，为确保数据准确性，将强制重新处理所有文件。")
            current_files_state = {}
            uploaded_files = []

    previous_files_state = load_processed_state()

    needs_rebuild = False
    if uploaded_files:
        if not previous_files_state:
            st.info("未找到之前的文档数据状态，将根据当前上传文件创建新的向量存储。")
            needs_rebuild = True
        elif current_files_state != previous_files_state:
            st.info("检测到上传文件与之前保存的数据状态不一致 (文件数量、名称或内容有变更)，将重新创建向量存储。")
            changed_files = [name for name, hash_val in current_files_state.items() if name not in previous_files_state or previous_files_state.get(name) != hash_val]
            removed_files_from_upload = [name for name in previous_files_state if name not in current_files_state]
            if changed_files: st.info(f"变动或新增文件: {', '.join(changed_files)}")
            if removed_files_from_upload: st.info(f"本次未上传但上次处理过的文件: {', '.join(removed_files_from_upload)}")
            needs_rebuild = True
        else:
            st.info("上传文件与之前保存的数据状态一致，将尝试加载现有的向量存储。")
            needs_rebuild = False
    elif previous_files_state:
        st.info("未上传新的文档，但检测到之前已处理的文档数据。正在尝试加载...")
        needs_rebuild = False
    else:
        st.warning("没有上传新的文档，也未找到之前的文档数据。请上传 PDF 文档以开始。")
        return None, False

    # 3. 根据判断结果执行加载或重建
    if needs_rebuild:
        if os.path.exists(PERSIST_DIRECTORY):
            st.info(f"正在清理旧的 Chroma 目录: {PERSIST_DIRECTORY}")
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                st.success("旧的 Chroma 目录已清理。")
            except Exception as cleanup_e:
                st.error(f"清理旧的 Chroma 目录失败: {cleanup_e}")
                logging.error(f"清理旧的 Chroma 目录失败: {cleanup_e}", exc_info=True)

        docs = []
        temp_dir = None
        all_splits = []

        try:
            temp_dir = tempfile.TemporaryDirectory()
            st.info(f"开始处理 {len(uploaded_files)} 个上传的文件 (用于 Chroma)...")

            for file in uploaded_files:
                temp_filepath = os.path.join(temp_dir.name, file.name)
                try:
                    file.seek(0)
                    with open(temp_filepath, "wb") as f:
                        f.write(file.getvalue())
                    loader = PyPDFLoader(temp_filepath)
                    file_docs = loader.load()
                    if not file_docs:
                         logging.warning(f"文件 {file.name} 加载后未产生文档。")
                         continue
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
                    splits = text_splitter.split_documents(file_docs)
                    all_splits.extend(splits)
                    logging.info(f"成功加载并分割了 {file.name}，产生 {len(splits)} 个片段。")
                except Exception as e:
                    st.error(f"加载或处理 PDF 文件 {file.name} 时出错: {e}")
                    logging.error(f"处理 {file.name} 时出错: {e}", exc_info=True)

            if not all_splits:
                st.warning("没有成功加载或分割任何上传的文档用于 Chroma。")
                return None, False

            try:
                st.info(f"正在创建 Chroma 向量存储并持久化到 {PERSIST_DIRECTORY} (可能需要一些时间)...")
                vectordb = Chroma.from_documents(
                    all_splits,
                    embeddings,
                    persist_directory=PERSIST_DIRECTORY
                )
                st.success(f"新的 Chroma 向量存储创建成功并已持久化到 {PERSIST_DIRECTORY}。")

                save_processed_state(current_files_state)
                st.success(f"已保存当前文件状态到 {PERSIST_STATE_FILE}。")

                chroma_retriever = vectordb.as_retriever(
                    search_type="mmr", # 使用 MMR 以增加多样性
                    search_kwargs={"k": 16}
                )
                st.success("Chroma 检索器配置成功 (MMR, k=16)。")
                return chroma_retriever, False

            except Exception as e:
                st.error(f"创建 Chroma 向量存储或检索器失败: {e}")
                logging.error("Chroma 创建失败:", exc_info=True)
                if "out of memory" in str(e).lower():
                     st.error("GPU 显存不足！")
                elif "meta tensor" in str(e):
                     st.error("加载模型时遇到 Meta Tensor 错误。")
                return None, False

        except Exception as e:
            st.error(f"在检索器配置过程中发生意外错误: {e}")
            logging.error("检索器配置失败:", exc_info=True)
            return None, False
        finally:
            if temp_dir:
                try:
                    temp_dir.cleanup()
                    logging.info("临时目录已清理。")
                except Exception as e:
                    logging.warning(f"无法清理临时目录: {e}")

    else: # 不需要重建，尝试从持久化目录加载
        if not os.path.exists(PERSIST_DIRECTORY):
             st.warning(f"持久化目录 {PERSIST_DIRECTORY} 不存在。请上传文件以创建数据库。")
             return None, False # 目录不存在，无法加载

        try:
            st.info(f"正在尝试从持久化目录加载 Chroma 向量存储: {PERSIST_DIRECTORY}...")
            vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            st.success("成功加载现有的 Chroma 向量存储。")
            chroma_retriever = vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 16}
            )
            st.success("Chroma 检索器加载成功 (MMR, k=16)。")
            was_loaded_from_disk = True
            return chroma_retriever, was_loaded_from_disk

        except Exception as e:
            st.error(f"从持久化目录 {PERSIST_DIRECTORY} 加载 Chroma 失败 ({e})。")
            logging.warning(f"Chroma 加载失败: {e}", exc_info=True)
            st.warning("加载现有数据失败，可能是数据损坏或版本不兼容。请尝试重新上传文件或点击侧边栏'清除所有数据缓存'按钮。")
            return None, False


# --- Streamlit 应用主逻辑 ---

# 1. 文件上传
uploaded_files = st.sidebar.file_uploader(
    label="上传 PDF 文件",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader_dashscope_chroma_cn_v2"
)

# 2. 配置检索器 (会检查文件变更并决定加载或创建)
retriever = None # 初始化为 None
loaded_from_disk = False # 初始化为 False
with st.spinner("正在处理上传的文档或加载现有的 Chroma 检索器并检查文件变更..."):
    retriever, loaded_from_disk = configure_retriever(uploaded_files)

# 如果检索器配置失败，停止应用 (或者给出提示)
if retriever is None:
    # 如果是因为没有上传文件且没有持久化数据，这是正常情况，不需要停止
    if not uploaded_files and not os.path.exists(PERSIST_DIRECTORY):
        st.info("请上传 PDF 文件以开始与文档对话。")
        # 可以选择停止，或者让应用继续运行但禁用聊天输入
        # st.stop()
    else:
        # 如果是其他错误（加载失败、创建失败等），显示错误并停止
        st.error("未能配置或加载 Chroma 文档检索器，无法继续。请检查上传的文件、错误日志或尝试清除缓存。")
        st.stop()

# 3. 初始化 LLM
llm = None # 初始化
if retriever is not None: # 仅在检索器成功配置后才初始化 LLM
    try:
        st.info(f"正在初始化 DashScope LLM: {MODEL_ID}...")
        llm = ChatOpenAI(
            model_name=MODEL_ID,
            openai_api_key=DASHSCOPE_API_KEY,
            openai_api_base=DASHSCOPE_API_BASE,
            temperature=0.5,
            max_tokens=1024,
        )
        st.success(f"DashScope LLM ({MODEL_ID}) 初始化成功。")
    except Exception as e:
        st.error(f"初始化 DashScope LLM ({MODEL_ID}) 失败: {e}")
        logging.error("DashScope LLM 初始化失败:", exc_info=True)
        if "api_key" in str(e).lower() or "authenticate" in str(e).lower():
            st.error("认证失败。请检查环境变量 DASHSCOPE_API_KEY 是否已正确设置且 Key 本身有效。")
        elif "base_url" in str(e).lower() or "connection" in str(e).lower():
             st.error(f"连接失败。请检查 Base URL ({DASHSCOPE_API_BASE}) 是否正确以及网络连接是否能访问该地址。")
        elif "model" in str(e).lower() or "not found" in str(e).lower():
             st.error(f"模型错误。请检查模型名称 '{MODEL_ID}' 是否是 DashScope 支持的有效模型。")
        else:
             st.error(f"请检查 API Key、模型名称 ('{MODEL_ID}')、Base URL 及网络连接。")
        st.stop()

# 4. 初始化聊天记录
msgs = StreamlitChatMessageHistory(key="rag_chat_messages_dashscope_chroma_cn_v2_persistent")

# 5. 定义 Prompt 模板
RESPONSE_TEMPLATE = """<s>[INST]
<<SYS>>
你是一个专业、耐心、且乐于助人的 AI 助手。请根据下面提供的上下文信息来回答用户的问题。
如果上下文信息足以回答问题，请直接依据上下文进行回答。
如果上下文信息不足以回答问题，或者问题与上下文无关，请明确告知用户上下文信息不足，并尝试根据你的通用知识进行回答（如果可能），同时说明这是基于通用知识而非文档内容。
回答应清晰、简洁，并使用中文。
<<SYS>>

从文档中检索到的相关上下文信息:
---
{context}
---

用户问题: {question}
[/INST]
AI 助手回答:
"""
PROMPT = PromptTemplate(template=RESPONSE_TEMPLATE, input_variables=["context", "question"])

# 6. 创建 RAG 问答链
qa_chain = None # 初始化
if llm is not None and retriever is not None: # 仅在 LLM 和 Retriever 都成功时创建
    try:
        st.info("正在创建 RAG 问答链 (使用 ChromaDB 和 DashScope LLM)...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        st.success("RAG 问答链准备就绪。")
    except Exception as e:
        st.error(f"创建 RAG QA 链失败: {e}")
        logging.error("RAG 链创建失败:", exc_info=True)
        st.stop()

# 7. 显示聊天界面和处理用户输入

# 确定初始消息
if len(msgs.messages) == 0 or st.sidebar.button("开始新对话", key="new_chat_button_dashscope_chroma_cn_v2_persistent"):
    msgs.clear()
    initial_message = "你好！请上传 PDF 文档，我可以根据文档内容回答你的问题。"
    if retriever is not None: # 如果检索器已就绪
        if loaded_from_disk:
            initial_message = f"已加载之前的文档数据（来自 `{PERSIST_DIRECTORY}`）。关于这些文档，有什么可以帮您的吗？"
        elif uploaded_files:
            initial_message = f"已处理您上传的文档并创建索引。关于这些文档，有什么可以帮您的吗？"
    msgs.add_ai_message(initial_message)

# 使用 Emojis 作为头像
avatars = {"human": "🧑‍💻", "ai": "🤖"}

# 显示历史消息
for msg in msgs.messages:
    st.chat_message(msg.type, avatar=avatars.get(msg.type)).write(msg.content)

# 处理用户输入
# 只有在 qa_chain 成功创建后才启用聊天输入框
chat_input_disabled = (qa_chain is None)
chat_input_placeholder = "请先上传文档并等待处理完成..." if chat_input_disabled else "就您的文档提出问题..."

if user_query := st.chat_input(placeholder=chat_input_placeholder, key="user_query_input_dashscope_chroma_cn_v2_persistent", disabled=chat_input_disabled):
    msgs.add_user_message(user_query)
    st.chat_message("human", avatar=avatars["human"]).write(user_query)

    with st.chat_message("ai", avatar=avatars["ai"]):
        placeholder = st.empty()
        placeholder.markdown("思考中 (正在使用 ChromaDB 检索和 DashScope LLM)...")
        try:
            # 调用 QA 链
            response = qa_chain.invoke({"query": user_query}) # 使用 invoke
            answer = response.get("result")

            if answer is None:
                 answer = "抱歉，未能生成回答。请尝试重新提问或检查模型状态。"
                 logging.error(f"QA chain for query '{user_query}' returned None result. Response: {response}")

            placeholder.empty() # 清除 "思考中"
            st.markdown(answer) # 显示最终答案
            msgs.add_ai_message(answer)

            # 显示上下文来源 (保持用于调试)
            retrieved_docs = response.get("source_documents", [])
            if retrieved_docs:
                with st.expander("查看本次查询检索到的上下文来源", expanded=False):
                     for i, doc in enumerate(retrieved_docs):
                        source = doc.metadata.get('source', '未知来源')
                        # 尝试提取更干净的文件名
                        source_display = os.path.basename(source) if source != '未知来源' else source
                        page = doc.metadata.get('page', '?')
                        if isinstance(page, int):
                             page += 1 # PDF 页码从 1 开始
                        st.write(f"**上下文块 {i+1}:** (来源: {source_display}, 页码: {page})")
                        st.caption(doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content)
                        st.write("---")
            else:
                logging.warning("问答链没有返回源文档信息。")

        except Exception as e:
            placeholder.empty()
            error_msg = f"处理您的问题时发生意外错误: {e}"
            st.error(error_msg)
            logging.error(f"处理查询 '{user_query}' 时出错:", exc_info=True)
            ai_error_message = f"抱歉，尝试回答您的问题时遇到了内部错误。" # 默认错误消息
            if "api_key" in str(e).lower() or "authenticate" in str(e).lower():
                 ai_error_message = "抱歉，尝试回答您的问题时遇到了 API 密钥认证错误。请联系管理员检查配置。"
            elif "rate limit" in str(e).lower():
                 ai_error_message = "抱歉，达到了 API 调用频率限制，请稍后再试。"
            elif "connection" in str(e).lower():
                  ai_error_message = "抱歉，尝试连接 DashScope API 时出错，请检查网络。"
            elif "context length" in str(e).lower():
                  ai_error_message = "抱歉，您的问题或检索到的上下文过长，超出了模型的处理限制。请尝试缩短问题或上传更小的文档。"
            # 在聊天记录中添加更友好的错误提示
            st.write(ai_error_message) # 也直接显示给用户
            msgs.add_ai_message(ai_error_message)


# --- 侧边栏 "关于" 部分 ---
about = st.sidebar.expander("关于此应用")
# 在关于部分动态显示是否加载了持久化数据
persistence_status = f'持久化到目录 `{PERSIST_DIRECTORY}`' # 显示相对路径
# 只有在检索器成功配置后才显示状态
if retriever is not None:
    if loaded_from_disk:
         persistence_status += " (已加载现有数据)"
    elif uploaded_files:
         persistence_status += " (已根据上传文件创建/更新)"
    else: # retriever is not None but not loaded_from_disk and no uploaded files
         persistence_status += " (已就绪，无文件上传)"
else:
     # 如果是因为没上传文件且无持久化数据，显示不同信息
     if not uploaded_files and not os.path.exists(PERSIST_DIRECTORY):
         persistence_status = f"持久化目录 `{PERSIST_DIRECTORY}` 尚未创建 (等待文件上传)"
     else:
         persistence_status = f"持久化目录 `{PERSIST_DIRECTORY}` 状态未知 (检索器配置失败)"


about.write(f"""
    这是一个基于 RAG (Retrieval-Augmented Generation) 的 AI 助手，可以与您上传的 PDF 文档进行对话。

    **关键特性:**
    *   支持上传 PDF 文档。
    *   使用 BGE 嵌入模型将文档内容转换为向量。
    *   使用 ChromaDB 存储和检索文档向量（**支持持久化与文件同步**）。
    *   利用 DashScope 的 {MODEL_ID} 模型根据检索到的信息生成回答。

    **技术栈:**
    *   **语言模型:** {MODEL_ID} (通过阿里云 DashScope)
    *   **嵌入模型:** BAAI BGE (langchain-huggingface, 在 {'GPU' if torch.cuda.is_available() else 'CPU'} 上运行)
    *   **检索器 & 向量库:** ChromaDB ({persistence_status})
    *   **框架:** Langchain & Streamlit

    **文件同步说明:**
    应用会计算上传文件的内容哈希值，并与之前保存的状态（在 `{PERSIST_STATE_FILE}`）进行比对。如果检测到您本次上传的文件集合与之前处理的不同（包括文件的添加、删除或内容修改），将自动清除旧数据（在 `{PERSIST_DIRECTORY}` 目录下），并根据当前上传的所有文件重新创建向量数据库。如果上传文件与之前一致，则快速加载已保存的数据。

    **注意:** 持久化数据保存在 `{PERSIST_DIRECTORY}` 目录中。请确保此目录已被添加到 `.gitignore` 文件，以避免将大量数据提交到版本控制。
""")


