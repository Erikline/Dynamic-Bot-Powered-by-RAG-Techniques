# RAG Chatbot Application (with ChromaDB Persistence & File Sync)

This is a Streamlit web application based on the Retrieval-Augmented Generation (RAG) architecture. It allows users to upload PDF documents and interact with their content through a chatbot interface. The application integrates ChromaDB as a vector database (with persistence), uses the Hugging Face BGE model for document embedding, and leverages the Deepseek-R1 model on Alibaba Cloud DashScope platform for generating responses. A key feature is its ability to detect changes in uploaded files and automatically manage the vector database synchronization accordingly.

## Overview

This project is an intelligent question-answering application based on the **Retrieval-Augmented Generation (RAG)** technique. Users can upload one or more documents in PDF format (especially suitable for competition regulations, etc.), and then ask questions about the content of these documents through a chat interface. The core of the application is a carefully designed **Competition Intelligent Customer Service** robot, which utilizes the **Deepseek-R1 large language model provided by Alibaba Cloud DashScope platform** and the **BAAI BGE embedding model** to understand and retrieve document content. It leverages a **ChromaDB vector database** to store and query document chunks, ultimately generating accurate, context-faithful answers.

## Key Features

*   **Multiple PDF Document Support**: Supports uploading and processing multiple PDF files simultaneously.
*   **RAG Core**: Integrates the Langchain framework to implement efficient document loading, splitting, embedding, retrieval, and generation workflows.
*   **Accurate Q&A**: Utilizes the BGE embedding model and ChromaDB for semantic retrieval to find the most relevant document chunks related to the user's question.
*   **Deepseek-R1 Powered**: Calls the Deepseek-R1 model via DashScope's OpenAI compatible API interface to generate natural, fluent, and context-based answers.
*   **Specialized Prompting**: Includes a carefully designed prompt template to guide the LLM in acting as a "Competition Intelligent Customer Service" role, emphasizing:
    *   **Strictly Context-Based**: Answers must originate from the provided document snippets, avoiding speculation and external knowledge interference (except in specific defined scenarios).
    *   **Awareness of Information Boundaries**: If the document lacks relevant information, the user must be clearly informed.
    *   **Structured Output**: Summarizes complex issues and presents them clearly, using formats like lists.
    *   **Prudent Use of General Knowledge**: Limits the use of general knowledge only when answering universal, non-competition-specific questions.
*   **GPU Acceleration**: Automatically detects and prioritizes the use of a CUDA GPU for loading the BGE embedding model, significantly improving processing speed. Falls back to CPU if no GPU is available.
*   **Streamlit Interface**: Built with Streamlit to provide a user-friendly web interface, including file uploading, chat history display, and real-time Q&A interaction.
*   **Chat History**: Maintains the chat history for the current session for user review.
*   **Context Source Display (Debugging)**: Optional expandable section to show the original document snippets used to generate the answer, including file name and page number.
*   **Error Handling**: Includes error capturing and prompts for common issues like API key errors, network connection problems, and model loading failures.
*   **Knowledge Base Persistence**: Saves the vector database to the local file system, allowing for quick loading upon application restart.
*   **File Change Synchronization**: Automatically detects file content changes using hashing and manages database rebuilding or loading to ensure the knowledge base is always current with the uploaded files.

## Tech Stack

*   **Large Language Model (LLM)**: Deepseek-R1 (via Alibaba Cloud DashScope Compatible API)
*   **Embedding Model (Embeddings)**: BAAI/bge-large-zh-v1.5 (loaded via `langchain-huggingface`)
*   **Vector Database (Vector Store)**: ChromaDB (supports local persistence)
*   **Core Frameworks**: Langchain, Streamlit
*   **PDF Processing**: PyPDFLoader
*   **Text Splitting**: RecursiveCharacterTextSplitter
*   **Environment Management**: dotenv
*   **Base Libraries**: PyTorch (for device detection and HuggingFace models), OpenAI (for DashScope compatible interface), Shutil (for directory operations), Hashlib (for file hashing), Pillow (for image handling)

## System Requirements

*   Python 3.9+
*   Pip (Python package manager)
*   **Recommended**: NVIDIA GPU (CUDA supported) for optimal embedding model processing performance. If no GPU is available, the application will use the CPU, but processing speed will be slower.
*   Sufficient RAM and disk space (depends on the size and number of PDFs).

## Installation and Deployment

1.  **Clone Repository**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    # For Windows:
    .\venv\Scripts\activate
    # For macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    Based on the `import` statements in your code, you need to install the following main libraries. It is recommended to create a `requirements.txt` file:

    ```txt
    # requirements.txt
    streamlit
    langchain
    langchain-community
    langchain-huggingface
    langchain-openai  # Note: Used for DashScope compatible API
    openai            # Required for DashScope compatible interface
    pypdf             # Dependency for PyPDFLoader
    chromadb
    torch             # Dependency for Huggingface Embeddings and device detection
    sentence-transformers # Dependency for Huggingface Embeddings
    pillow            # PIL for image handling
    python-dotenv
    # Based on your specific environment and dependency versions, adjustments might be needed.
    ```

    Then run the installation command:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: When installing `torch`, please refer to the [PyTorch website](https://pytorch.org/) for instructions on selecting the correct version command based on your system and whether you need GPU support. For example, a version with CUDA 12.1 support:*
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Configure Environment Variables**:
    *   Obtain your Alibaba Cloud DashScope API Key. Reference: [Get API Key](https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key)
    *   Create a file named `.env` in the root directory of the project.
    *   Add your API Key to the `.env` file:
        ```env
        DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ```
    *   (Optional) Modify the `FAVICON_PATH` and `LOGO_PATH` in the code to the correct paths for your local icon and logo images, or remove the relevant code lines if you don't use custom images.
    *   (Optional) Modify `PERSIST_DIRECTORY` in the code if you prefer a different location for ChromaDB data. Ensure the user running the script has write permissions for this directory. *Note: The provided code uses an absolute Windows path (`D:\Desktop\...`). It's generally recommended to use relative paths like `./chroma_db_streamlit` for better portability across different environments.*

## Usage

1.  **Launch Application**:
    In the project root directory, ensure your virtual environment is activated and run the Streamlit command:
    ```bash
    streamlit run <your_script_name>.py
    ```
    (Replace `<your_script_name>.py` with the actual filename where you saved the code).

2.  **Upload Files**:
    Once the application is running, in the sidebar of the page opened in your browser, click "Browse files" or drag and drop the PDF documents you want to query.

3.  **Wait for Processing**:
    The application will automatically process the uploaded PDF files. It will check for file changes. If changes are detected or no previous data exists, it will perform text splitting, generate embedding vectors, and store them in ChromaDB. This process may take some time, especially during the first processing or with large/numerous files, or when a rebuild is triggered. Status updates will be shown in the sidebar and main interface. If no changes are detected and data exists, it will load the existing database quickly.

4.  **Start Asking Questions**:
    After document processing/loading is complete, the application will show a welcome message. Type your question about the document content in the chat input box at the bottom of the page and press Enter or click the send button.

5.  **View Answer**:
    The application will retrieve relevant information, use the Deepseek-R1 model to generate an answer, and display it in the chat interface.

6.  **View Context (Optional)**:
    Below the AI's answer, there is usually an expandable section ("View retrieved context sources"). Click it to see the specific document snippets used to generate that answer, including their source and page number.

7.  **Start New Conversation**:
    If you want to clear the current chat history and start a new conversation, you can click the "Start new conversation" button in the sidebar.

8.  **Clear All Data**:
    The "Clear all data cache (including ChromaDB)" button in the sidebar will delete the locally saved ChromaDB data and the file state tracking file, as well as clear Streamlit's resource cache. Use this to force a complete reprocessing of files the next time you upload them or restart the app.

## Notes and Troubleshooting

*   **API Key Error**: Ensure your `DASHSCOPE_API_KEY` in the `.env` file is correct and valid. Check if the key is mistyped, expired, or if your account has a balance issue.
*   **Network Connection**: Confirm that your network environment can access the DashScope API endpoint (`https://dashscope.aliyuncs.com/compatible-mode/v1`).
*   **Model Name**: Verify that `MODEL_ID = "deepseek-r1"` in the code is a valid model name supported by DashScope.
*   **GPU Out of Memory (OOM)**: If you encounter an OOM error while processing documents or loading the embedding model, it might be due to insufficient GPU memory. Try:
    *   Closing other programs that are using GPU memory.
    *   Reducing the number or size of PDF files uploaded simultaneously.
    *   (Code Modification) If the issue persists, you can force the `device` to `'cpu'` in the `configure_retriever` function, although this will reduce performance significantly.
*   **Meta Tensor Error**: If you encounter a `meta tensor` related error when loading the Hugging Face model, try clearing the Hugging Face model cache (usually located at `~/.cache/huggingface/hub` or `C:\Users\<User>\.cache\huggingface\hub`) and restart the application.
*   **PDF Processing Failure**: Some PDFs might not be parsed correctly by `PyPDFLoader` due to formatting issues. Try using a different PDF or check if the file is corrupted.
*   **Performance**: Processing large or numerous PDF documents, or triggering a database rebuild, can take a significant amount of time for embedding and indexing, especially when running on a CPU. Please be patient.


## Acknowledgements

*   Langchain ([https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain))
*   Streamlit ([https://github.com/streamlit/streamlit](https://github.com/streamlit/streamlit))
*   ChromaDB ([https://github.com/chroma-core/chroma](https://github.com/chroma-core/chroma))
*   Hugging Face Transformers ([https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)) & Embeddings ([https://huggingface.co/BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5))
*   Alibaba Cloud DashScope ([https://www.aliyun.com/product/bailian](https://www.aliyun.com/product/bailian)) & Deepseek-R1 Model

## Experience at that moment
* Welcome to the RAG Bot experience! For more details, visit: https://dynamic-bot-powered-by-rag-techniques.streamlit.app/
---
