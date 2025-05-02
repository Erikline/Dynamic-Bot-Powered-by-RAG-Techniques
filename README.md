# RAG Chatbot 应用（基于 ChromaDB 持久化与文件同步）

这是一个基于检索增强生成 (RAG) 架构的 Streamlit Web 应用，允许用户上传 PDF 文档并与其内容进行对话。该应用集成了 ChromaDB 作为向量数据库（支持持久化），使用 Hugging Face 的 BGE 模型进行文档嵌入，并利用阿里云 DashScope 平台上的 Deepseek-R1 模型生成回答。其核心特性是能够检测上传文件的变更，并据此自动管理向量数据库的同步。

## 功能特性

*   **PDF 文档上传:** 通过侧边栏轻松上传一个或多个 PDF 文件。
*   **RAG 架构:** 利用外部知识库增强大型语言模型的回答能力。
*   **文档处理:** 使用 `PyPDFLoader` 加载 PDF，`RecursiveCharacterTextSplitter` 进行文本分块。
*   **高效嵌入:** 采用 `BAAI/bge-large-zh-v1.5` 嵌入模型，并自动检测并利用 GPU (如果可用) 进行加速。
*   **向量存储与检索:** 使用 `ChromaDB` 作为向量数据库，通过配置的 MMR 检索器 (`top-k=16`) 进行高效的相关文档片段检索。
*   **知识库持久化:** 将向量数据库保存到本地文件系统，以便应用重启后快速加载，无需重新处理文件。
*   **文件变更同步:** 通过计算文件内容哈希值，智能检测上传文件的变更（新增、删除、修改）。如果文件集合发生变更，自动清除旧数据并重新处理所有当前文件以重建向量数据库，确保知识库始终与最新文件同步。
*   **大模型交互:** 集成阿里云 DashScope 平台上的 `Deepseek-R1` 模型（通过 OpenAI 兼容模式）进行问答。
*   **Prompt 工程:** 使用精心设计的 Prompt 模板指导大模型生成忠实于文档上下文、结构清晰的回答。
*   **聊天界面:** 提供直观的 Streamlit 聊天界面，支持历史消息展示。
*   **源文档追溯:** 在回答下方可选显示检索到的上下文来源及其页码。
*   **缓存管理:** 使用 Streamlit 的 `st.cache_resource` 缓存检索器，提高效率；提供按钮清除缓存和持久化数据。

## 技术栈

*   **前端 & 应用框架:** Streamlit
*   **核心 RAG 框架:** Langchain
*   **大型语言模型 (LLM):** Deepseek-R1 (通过阿里云 DashScope API)
*   **嵌入模型:** BAAI/bge-large-zh-v1.5 (HuggingFace Embeddings)
*   **向量数据库:** ChromaDB (支持本地持久化)
*   **文件处理:** PyPDFLoader
*   **文本分块:** RecursiveCharacterTextSplitter
*   **实用工具:** python-dotenv, torch, shutil, hashlib, Pillow

## 环境要求

*   Python 3.9+
*   GPU (可选，但强烈推荐用于加速 Embedding 模型)
*   阿里云 DashScope API Key

## 设置步骤

1.  **克隆或下载代码:**
    ```bash
    git clone <你的代码仓库地址> # 如果代码在 Git 仓库
    # 或者手动下载 Python 脚本文件
    ```

2.  **安装依赖库:**
    建议使用虚拟环境 (如 `venv` 或 `conda`)。
    ```bash
    # 创建并激活虚拟环境 (示例使用 venv)
    python -m venv .venv
    source .venv/bin/activate # macOS/Linux
    # .venv\Scripts\activate # Windows

    # 安装所需的库
    pip install -U streamlit langchain langchain-community langchain-huggingface langchain-openai chromadb python-dotenv torch Pillow
    # 注意：如果你没有 GPU 或不想使用 GPU，安装 torch 时可能需要指定版本或平台。
    # 例如，仅CPU版本: pip install torch torchvision torchaudio cpuonly
    ```

3.  **配置阿里云 DashScope API Key:**
    *   在项目根目录下创建一个名为 `.env` 的文件。
    *   在 `.env` 文件中添加你的 DashScope API Key：
        ```dotenv
        DASHSCOPE_API_KEY="你的_实际_DashScope_API_Key"
        ```
    *   你可以从阿里云百炼控制台获取 API Key：[获取API Key](https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key)

4.  **配置本地路径 (重要):**
    打开 Python 脚本文件，根据你的实际文件位置修改以下路径：
    *   `FAVICON_PATH`：浏览器 Tab 图标路径。
    *   `LOGO_PATH`：侧边栏 Logo 图标路径。
    *   `PERSIST_DIRECTORY`：ChromaDB 数据将要存储的本地目录路径。请确保运行脚本的用户有权限在该路径下创建和写入文件。虽然你的代码使用了绝对路径，通常推荐使用相对路径如 `./chroma_db_streamlit` 以增强可移植性。

5.  **准备 PDF 文件:**
    将你想要应用处理的 PDF 文档准备好。你将在 Streamlit 界面的侧边栏上传这些文件。

## 运行应用

在已激活虚拟环境的终端中，导航到脚本所在的目录，然后运行：

```bash
streamlit run your_script_name.py # 将 your_script_name.py 替换为你的 Python 文件名
```

应用将在你的浏览器中打开一个新的标签页。

## 使用说明

1.  **上传 PDF:** 在 Streamlit 页面的侧边栏使用 "上传 PDF 文件" 按钮上传你想要处理的竞赛文档。你可以上传多个文件。
2.  **文档处理:** 应用会自动检测你上传的文件。
    *   如果是第一次上传，或者你上传的文件与之前保存的状态不一致（文件名、数量或内容有变），应用会开始处理文档（加载、分块、嵌入、创建向量数据库）并将数据保存到 `PERSIST_DIRECTORY` 指定的目录。这个过程可能需要一些时间，特别是在 CPU 上处理大量文件时。
    *   如果你上传的文件与之前保存的状态一致，或者你未上传文件但 `PERSIST_DIRECTORY` 目录存在且包含有效数据，应用会快速加载现有的向量数据库，从而加快启动速度。
3.  **开始聊天:** 文档处理完成后，应用会显示欢迎消息，你可以在底部的输入框中输入关于你上传文档的问题。
4.  **查看上下文:** 发送问题后，应用会显示回答。你可以点击 "查看本次查询检索到的上下文来源" 展开器，查看大模型在生成回答时参考的文档片段及其来源页码。
5.  **文件同步:** 如果后续你修改了之前上传的 PDF 文件，或者添加/删除了文件，只需再次通过侧边栏上传 **所有** 你当前希望纳入知识库的文件（包括未修改的）。应用会自动检测变更，并重建数据库。
6.  **清除数据:** 侧边栏的 "清除所有数据缓存 (包括 ChromaDB)" 按钮会删除本地保存的 ChromaDB 数据和文件状态记录，并清除 Streamlit 缓存。使用此功能将强制应用在下次运行时重新处理所有上传的文件。


## 致谢

*   Langchain ([https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain))
*   Streamlit ([https://github.com/streamlit/streamlit](https://github.com/streamlit/streamlit))
*   ChromaDB ([https://github.com/chroma-core/chroma](https://github.com/chroma-core/chroma))
*   Hugging Face Transformers ([https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)) & Embeddings ([https://huggingface.co/BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5))
*   阿里云 DashScope ([https://www.aliyun.com/product/bailian](https://www.aliyun.com/product/bailian)) & Deepseek-R1 Model

## 📄 许可证 (License)

本项目根据 **[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)** 条款进行许可。

## 即时体验
* 欢迎访问该链接进行 RAG Bot 体验！详情访问：[Dynamic Bot Powered by RAG Techniques](https://dynamic-bot-powered-by-rag-techniques.streamlit.app/)
*  ![Alt text](https://github.com/Erikline/Dynamic-Bot-Powered-by-RAG-Techniques/blob/main/Streamlit%20Web.png)

