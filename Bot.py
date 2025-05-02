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
import torch # <--- 导入 torch 用于设备检查
import shutil # <--- 导入 shutil 用于目录操作
import hashlib # <--- 导入 hashlib 用于计算文件哈希

# 加载环境变量 (确保 .env 文件存在或环境变量已设置)
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
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

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
    st.error("未找到 DashScope API 密钥。请设置环境变量 DASHSCOPE_API_KEY 或在 .env 文件中配置。")
    st.info("如何获取API Key: https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key")
    st.stop()

# --- 模型配置 ---
MODEL_ID = "deepseek-r1" # 使用 DashScope 的模型名称

# --- 持久化配置 ---
PERSIST_DIRECTORY = r"D:\Desktop\程序\03-构建动态LLM-Bot模型代码\chroma_db_streamlit" # <--- ChromaDB 数据将保存到这个目录
PERSIST_STATE_FILE = os.path.join(PERSIST_DIRECTORY, "processed_files_state.json") # <--- 用于记录文件状态的json文件

# --- 应用设置 ---
APP_TITLE = f"使用 {MODEL_ID} 和 基于 BGE 词嵌入 的 ChromaDB 的 RAG 应用 (支持持久化与文件同步)"
# !! 请确保下面的路径对你的环境是正确的 !!
FAVICON_PATH = r"D:\Desktop\程序\03-构建动态LLM-Bot模型代码\Bot.png" # 请替换为你的实际路径
LOGO_PATH = r"D:\Desktop\程序\03-构建动态LLM-Bot模型代码\icon.png"   # 请替换为你的实际路径

try:
    favicon = Image.open(FAVICON_PATH)
    st.set_page_config(page_title=APP_TITLE, page_icon=favicon, layout="wide")
except FileNotFoundError:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.warning(f"在路径 {FAVICON_PATH} 未找到 Favicon 图标。跳过页面图标设置。")
except Exception as e:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.warning(f"无法加载 Favicon 图标: {e}")

# --- Streamlit 侧边栏 ---
try:
    st.sidebar.image(LOGO_PATH, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning(f"在路径 {LOGO_PATH} 未找到 Logo 图片。")
except Exception as e:
     st.sidebar.warning(f"无法加载 Logo 图片: {e}")

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
    # Rewind the file pointer to the beginning
    uploaded_file.seek(0)
    # Read the file in chunks to handle large files
    while True:
        data = uploaded_file.read(8192) # Read in 8KB chunks
        if not data:
            break
        sha256.update(data)
    # Reset file pointer again for subsequent reads (e.g., by loader)
    uploaded_file.seek(0)
    return sha256.hexdigest()

def load_processed_state():
    """从状态文件加载之前处理的文件状态（文件名和哈希值）。"""
    if os.path.exists(PERSIST_STATE_FILE):
        try:
            with open(PERSIST_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"加载文件状态文件失败: {e}")
            return {} # 加载失败则返回空字典
    return {}

def save_processed_state(state_dict):
    """将当前文件状态（文件名和哈希值）保存到状态文件。"""
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    try:
        with open(PERSIST_STATE_FILE, 'w') as f:
            json.dump(state_dict, f, indent=4)
    except Exception as e:
        logging.error(f"保存文件状态文件失败: {e}", exc_info=True)


# --- 核心 RAG 功能 (configure_retriever - 应用 GPU 修正 & 持久化 & 文件同步) ---
@st.cache_resource(ttl="2h") # 缓存2小时
# 修改函数签名和返回值，返回 (retriever, boolean_was_loaded)
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
        # 检查 CUDA 是否可用，如果可用则使用 GPU，否则回退到 CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"检测到可用设备: {device}，将用于嵌入模型加载。")
        if device == 'cuda':
             # 尝试获取 GPU 型号，如果失败则忽略
             try:
                st.info(f"GPU 型号: {torch.cuda.get_device_name(0)}")
             except Exception:
                st.warning("无法获取 GPU 型号信息。")

        # 定义模型加载参数，明确指定设备
        model_kwargs = {'device': device}
        # BGE 模型推荐对嵌入进行归一化
        encode_kwargs = {'normalize_embeddings': True}

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5", # 确保模型名称正确
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
            st.error("加载模型时遇到 Meta Tensor 错误。请尝试清理 Hugging Face 模型缓存（通常在 ~/.cache/huggingface/hub 或 C:\\Users\\<用户>\\.cache\\huggingface\\hub），然后重启应用。如果问题持续，请检查 PyTorch 和 transformers 库版本。")
        else:
             st.error("请检查模型名称、网络连接或系统资源。")
        return None, False # Embedding 模型失败，无法继续，返回None和False

    # 2. 检查文件状态并决定加载或重建
    current_files_state = {}
    if uploaded_files:
        st.info("正在计算上传文件的哈希值以检查是否有变更...")
        try:
            # 计算当前上传文件的状态 (文件名: 哈希值)
            for file in uploaded_files:
                current_files_state[file.name] = calculate_file_hash(file)
            st.success(f"已计算 {len(uploaded_files)} 个文件的哈希值。")
        except Exception as e:
            st.error(f"计算文件哈希值失败: {e}")
            logging.error("计算哈希失败:", exc_info=True)
            # 如果计算哈希失败，无法安全地判断文件是否变更，此时稳妥起见强制重建
            st.warning("文件哈希计算失败，为确保数据准确性，将强制重新处理所有文件。")
            current_files_state = {} # 置空，确保不匹配旧状态，强制重建
            uploaded_files = [] # 清空uploaded_files列表，以便下一步走重建流程

    # 加载之前保存的文件状态
    previous_files_state = load_processed_state()

    # 判断是否需要重新构建向量数据库
    # 需要重建的情况：
    # 1. 没有上传文件 (uploaded_files 为空)，但之前保存的状态不为空（说明用户这次没传文件，但上次处理过） -> 尝试加载旧数据
    # 2. 上传了文件 (uploaded_files 非空)，但之前没有保存的状态文件 或者 当前文件状态与之前保存的状态不一致 -> 重建
    needs_rebuild = False
    if uploaded_files: # 如果用户上传了文件
        if not previous_files_state:
            st.info("未找到之前的文档数据状态，将根据当前上传文件创建新的向量存储。")
            needs_rebuild = True
        elif current_files_state != previous_files_state:
            st.info("检测到上传文件与之前保存的数据状态不一致 (文件数量、名称或内容有变更)，将重新创建向量存储。")
            # 详细说明哪些文件变了（可选）
            changed_files = [name for name, hash_val in current_files_state.items() if name not in previous_files_state or previous_files_state.get(name) != hash_val]
            removed_files_from_upload = [name for name in previous_files_state if name not in current_files_state]
            if changed_files: st.info(f"变动或新增文件: {', '.join(changed_files)}")
            if removed_files_from_upload: st.info(f"本次未上传但上次处理过的文件: {', '.join(removed_files_from_upload)}")

            needs_rebuild = True
        else:
            st.info("上传文件与之前保存的数据状态一致，将尝试加载现有的向量存储。")
            needs_rebuild = False # 状态一致，不需要重建
    elif previous_files_state: # 如果没有上传文件，但有旧状态（意味着之前处理过文件）
        st.info("未上传新的文档，但检测到之前已处理的文档数据。正在尝试加载...")
        needs_rebuild = False # 尝试加载旧数据
    else: # 既没有上传文件，也没有旧状态
        st.warning("没有上传新的文档，也未找到之前的文档数据。请上传 PDF 文档以开始。")
        return None, False # 无法继续，返回 None和False

    # 3. 根据判断结果执行加载或重建
    if needs_rebuild:
        # 清除旧的 ChromaDB 目录和状态文件
        if os.path.exists(PERSIST_DIRECTORY):
            st.info(f"正在清理旧的 Chroma 目录: {PERSIST_DIRECTORY}")
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                st.success("旧的 Chroma 目录已清理。")
            except Exception as cleanup_e:
                st.error(f"清理旧的 Chroma 目录失败: {cleanup_e}")
                logging.error(f"清理旧的 Chroma 目录失败: {cleanup_e}", exc_info=True)

        # 处理上传文件并创建新的 ChromaDB
        docs = []
        temp_dir = None
        all_splits = []

        try:
            # 3.1. 处理上传文件用于 Chroma
            temp_dir = tempfile.TemporaryDirectory()
            st.info(f"开始处理 {len(uploaded_files)} 个上传的文件 (用于 Chroma)...")

            for file in uploaded_files:
                # 注意：calculate_file_hash 已经把文件指针重置了
                temp_filepath = os.path.join(temp_dir.name, file.name)
                try:
                    # 再次确保文件指针在开始，虽然calculate_file_hash已做，这里是双重保险
                    file.seek(0)
                    with open(temp_filepath, "wb") as f:
                        f.write(file.getvalue())
                    loader = PyPDFLoader(temp_filepath)
                    file_docs = loader.load()
                    if not file_docs:
                         logging.warning(f"文件 {file.name} 加载后未产生文档。")
                         continue
                    # 使用之前调试确认的较优参数
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
                    splits = text_splitter.split_documents(file_docs)
                    all_splits.extend(splits)
                    logging.info(f"成功加载并分割了 {file.name}，产生 {len(splits)} 个片段。")
                except Exception as e:
                    st.error(f"加载或处理 PDF 文件 {file.name} 时出错: {e}")
                    logging.error(f"处理 {file.name} 时出错: {e}", exc_info=True)

            if not all_splits:
                st.warning("没有成功加载或分割任何上传的文档用于 Chroma。")
                return None, False # 没有文档处理成功，无法创建向量库

            # 3.2. 创建 Chroma 向量存储并持久化
            try:
                st.info(f"正在创建 Chroma 向量存储并持久化到 {PERSIST_DIRECTORY} (可能需要一些时间)...")
                # from_documents 会在指定的 persist_directory 创建并保存数据库
                vectordb = Chroma.from_documents(
                    all_splits,
                    embeddings, # <--- 传入前面创建的embeddings实例
                    persist_directory=PERSIST_DIRECTORY
                )
                st.success(f"新的 Chroma 向量存储创建成功并已持久化到 {PERSIST_DIRECTORY}。")

                # 保存当前文件状态
                save_processed_state(current_files_state)
                st.success("已保存当前文件状态。")

                # 获取检索器
                chroma_retriever = vectordb.as_retriever(
                    search_type="sim",
                    search_kwargs={"k": 16}
                )
                st.success("Chroma 检索器配置成功。")
                return chroma_retriever, False # 成功创建并持久化后返回检索器和False

            except Exception as e:
                st.error(f"创建 Chroma 向量存储或检索器失败: {e}")
                logging.error("Chroma 创建失败:", exc_info=True)
                # 添加显存不足的提示等
                if "out of memory" in str(e).lower():
                     st.error("GPU 显存不足！请尝试关闭其他占用显存的程序，或在代码中将 device 改为 'cpu'。")
                elif "meta tensor" in str(e):
                     st.error("加载模型时遇到 Meta Tensor 错误。请尝试清理 Hugging Face 模型缓存，然后重启应用。")
                else:
                    pass # st.error 已在上面显示了通用错误
                return None, False # 创建失败则返回 None和False

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
        try:
            st.info(f"正在尝试从持久化目录加载 Chroma 向量存储: {PERSIST_DIRECTORY}...")
            # Chroma 会自动加载如果数据存在
            # 注意：这里加载时也需要 embedding_function 参数
            vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            st.success("成功加载现有的 Chroma 向量存储。")
            chroma_retriever = vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 16}
            )
            st.success("Chroma 检索器加载成功。")
            was_loaded_from_disk = True # 设置加载标志为True
            return chroma_retriever, was_loaded_from_disk # 成功加载，返回检索器和True

        except Exception as e:
            st.error(f"从持久化目录加载 Chroma 失败 ({e})。")
            logging.warning(f"Chroma 加载失败: {e}", exc_info=True)
            # 加载失败时，即使之前判断不需要重建，现在也无法使用旧数据
            st.warning("加载现有数据失败，请尝试重新上传文件或清除缓存。")
            return None, False # 加载失败，返回 None和False


# --- Streamlit 应用主逻辑 ---

# 1. 文件上传
uploaded_files = st.sidebar.file_uploader(
    label="上传 PDF 文件",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader_dashscope_chroma_cn_v2" # 添加 key 确保唯一性
)

# 2. 配置检索器 (会检查文件变更并决定加载或创建)
with st.spinner("正在处理上传的文档或加载现有的 Chroma 检索器并检查文件变更..."):
    # 调用函数并接收两个返回值
    retriever, loaded_from_disk = configure_retriever(uploaded_files)

# 如果检索器配置失败，停止应用
if retriever is None:
    st.error("未能配置或加载 Chroma 文档检索器，无法继续。请检查上传的文件、错误日志或尝试清除缓存。")
    st.stop()

# 3. 初始化 LLM (这部分不受文件上传和持久化影响，每次运行 Streamlit 都会执行)
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
# 确保聊天记录的 key 足够独特，以免与其他应用的聊天记录混合
msgs = StreamlitChatMessageHistory(key="rag_chat_messages_dashscope_chroma_cn_v2_persistent")

# 5. 定义 Prompt 模板 (保持不变)
RESPONSE_TEMPLATE = """<s>[INST]
<<SYS>>
你的角色是一位 **竞赛智能客服** 机器人，专注于解读用户上传的 **特定竞赛规程 PDF 文档**。
你的核心任务是：基于下方提供的、从这些 **特定文档中检索到的上下文片段**，以**清晰、结构化、高度准确**的方式回答用户关于 **这些文档内容** 的问题。

指导原则：

1.  **铁证如山，绝不臆断:** 你的回答必须 **严格基于且完全源自** 以下提供的上下文信息。任何超出这些特定上下文的内容，包括但不限于你自身的通用知识、其他赛事信息、网络数据或个人推断，都**绝对禁止**使用。回答中的每一句话都应能在提供的上下文中找到直接或间接的明确支撑。

2.  明确信息边界，坦诚缺失：如果用户的问题涉及到 **本次竞赛的具体细节**，但所提供的文档片段中 **并未包含** 相关信息，你必须明确告知用户，无法基于现有文档回答该特定问题。例如，可以这样回答：“根据当前提供的竞赛文档信息，我暂时无法找到关于‘[用户问题关键点]’的具体说明。” 关键在于：**绝不** 能自行编造或猜测任何 **竞赛相关** 的信息。

3.  **提炼关键，结构化呈现 (仅限上下文内):** 在确保严格遵守原则1和2的前提下，当你能从上下文中找到答案时，请**仔细分析并提炼核心信息点**。如果上下文包含多个相关片段，尝试将信息进行**逻辑整合与归纳**。对于复杂问题，可以考虑使用要点、列表等方式结构化地呈现答案，使其既全面又有条理，体现对 **所提供信息** 的深度理解。

4.  审慎运用通用知识：只有当用户的问题 **明显** 是关于一个 **普遍性、非特定于本次竞赛** 的概念或知识（例如：“什么是自然语言处理？”、“请解释一下聊天机器人。”、“数据预处理有哪些通用建议？”），**并且** 问题内容 **不涉及** 上传的PDF文档 **特有的** 规则、安排或要求时，你 **可以** 运用你的通用知识库来提供背景信息或解释。请注意，这类回答应作为一般性知识提供，并明确（如果需要）这不代表本次竞赛的具体规定，除非文档中明确支持。

5.  沟通清晰友好：请以专业、清晰、友好的方式进行回答。在确保信息准确（特别是竞赛相关信息严格依据文档）的前提下，可以采用比纯粹列举事实稍微灵活一些的语言风格，使回答更易于理解。如果基于文档的回答内容较复杂，可以使用项目符号或列表等形式，使其结构更清晰。

现在，请仔细审阅下方提供的上下文信息，然后回答用户的问题。
<<SYS>>

从所提供的竞赛PDF文档中检索到的相关上下文信息:
---
{context}
---

请严格遵守以上原则进行回答。

用户问题: {question}
[/INST]
竞赛智能客服助手回答:
"""

PROMPT = PromptTemplate(template=RESPONSE_TEMPLATE, input_variables=["context", "question"])


# 6. 创建 RAG 问答链
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

# 根据是否是第一次运行或点击了“新对话”，以及数据是否加载自磁盘来显示初始消息
if len(msgs.messages) == 0 or st.sidebar.button("开始新对话", key="new_chat_button_dashscope_chroma_cn_v2_persistent"):
    msgs.clear()
    # 使用从configure_retriever返回的loaded_from_disk标志
    if loaded_from_disk:
         initial_message = f"已加载之前的文档数据。关于这些文档（使用 ChromaDB 检索和 DashScope LLM），有什么可以帮您的吗？"
    elif uploaded_files: # 如果有上传文件且不是加载的，那就是新建或重建了
         initial_message = f"已处理您上传的文档并创建索引。关于这些文档（使用 ChromaDB 检索和 DashScope LLM），有什么可以帮您的吗？"
    else: # 没有上传文件，也没有加载旧数据 (理论上前面应该stop了，这里作为安全网)
         initial_message = "请上传 PDF 文档以开始对话。"

    msgs.add_ai_message(initial_message)


# 使用 Emojis 作为头像
avatars = {"human": "🧑‍💻", "ai": "🤖"}

# 显示历史消息
for msg in msgs.messages:
    st.chat_message(msg.type, avatar=avatars.get(msg.type)).write(msg.content)

# 处理用户输入
if user_query := st.chat_input(placeholder="就您的文档提出问题...", key="user_query_input_dashscope_chroma_cn_v2_persistent"):
    msgs.add_user_message(user_query)
    st.chat_message("human", avatar=avatars["human"]).write(user_query)

    with st.chat_message("ai", avatar=avatars["ai"]):
        placeholder = st.empty()
        placeholder.markdown("思考中 (正在使用 ChromaDB 检索和 DashScope LLM)...")
        try:
            # 调用 QA 链
            response = qa_chain.invoke({"query": user_query})
            answer = response.get("result")

            if answer is None:
                 answer = "抱歉，未能生成回答。请尝试重新提问或检查模型状态。"
                 logging.error(f"QA chain for query '{user_query}' returned None result. Response: {response}")

            placeholder.empty()
            st.markdown(answer)
            msgs.add_ai_message(answer)

            # --- > 添加显示上下文的调试代码（保持，用于验证检索效果） <---
            retrieved_docs = response.get("source_documents", [])
            if retrieved_docs:
                with st.expander("查看本次查询检索到的上下文来源", expanded=False): # 默认折叠
                     for i, doc in enumerate(retrieved_docs):
                        source = doc.metadata.get('source', '未知来源')
                        page = doc.metadata.get('page', '?')
                        if isinstance(page, int):
                             page += 1 # PDF 页码从 1 开始
                        st.write(f"**上下文块 {i+1}:** (来源: {source}, 页码: {page})")
                        # 显示部分内容即可，避免过长
                        st.caption(doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content)
                        st.write("---")
            else:
                logging.warning("问答链没有返回源文档信息。")
            # --- > 调试代码结束 <---


        except Exception as e:
            placeholder.empty()
            error_msg = f"处理您的问题时发生意外错误: {e}"
            st.error(error_msg)
            logging.error(f"处理查询 '{user_query}' 时出错:", exc_info=True)
            if "api_key" in str(e).lower() or "authenticate" in str(e).lower():
                 msgs.add_ai_message("抱歉，尝试回答您的问题时遇到了 API 密钥认证错误。请联系管理员检查配置。")
            elif "rate limit" in str(e).lower():
                 msgs.add_ai_message("抱歉，达到了 API 调用频率限制，请稍后再试。")
            elif "connection" in str(e).lower():
                  msgs.add_ai_message("抱歉，尝试连接 DashScope API 时出错，请检查网络。")
            else:
                 msgs.add_ai_message(f"抱歉，尝试回答您的问题时遇到了内部错误。错误详情: {str(e)[:100]}...")


# --- 侧边栏 "关于" 部分 ---
about = st.sidebar.expander("关于此应用")
# 在关于部分动态显示是否加载了持久化数据
persistence_status = f'持久化到目录 `{PERSIST_DIRECTORY}`'
if retriever is not None: # 只有在检索器成功配置后才显示状态
    if loaded_from_disk:
         persistence_status += " (已加载现有数据)"
    elif uploaded_files:
         persistence_status += " (已根据上传文件创建/更新)"
    else: # retriever is not None but not loaded_from_disk and no uploaded files - this case should ideally not happen if retriever is not None
         persistence_status += " (已就绪，无文件上传)"
else:
     persistence_status = "持久化状态未知 (检索器配置失败)"


about.write(f"""
    这是一个基于 RAG (Retrieval-Augmented Generation) 的 AI 助手，可以与您上传的 PDF 文档进行对话。

    **关键特性:**
    *   支持上传 PDF 文档。
    *   使用 BGE 嵌入模型将文档内容转换为向量。
    *   使用 ChromaDB 存储和检索文档向量（**支持持久化与文件同步**）。
    *   利用 DashScope 的 {MODEL_ID} 模型根据检索到的信息生成回答。

    **技术栈:**
    *   **语言模型:** {MODEL_ID} (通过阿里云 DashScope)
    *   **嵌入模型:** BAAI BGE (langchain-huggingface, 在 {'GPU' if torch.cuda.is_available() else 'CPU'} 上运行) # <-- 动态显示设备
    *   **检索器 & 向量库:** ChromaDB ({persistence_status})
    *   **框架:** Langchain & Streamlit

    **文件同步说明:**
    应用会计算上传文件的内容哈希值，并与之前保存的状态进行比对。如果检测到您本次上传的文件集合与之前处理的不同（包括文件的添加、删除或内容修改），将自动清除旧数据，并根据当前上传的所有文件重新创建向量数据库，以确保检索的准确性。如果上传文件与之前一致，则快速加载已保存的数据。
""")