# Agentic RAG Chatbot with MCP Integration

An advanced Agent-based Retrieval-Augmented Generation (RAG) chatbot using Model Context Protocol (MCP) for multi-agent communication. The system processes multiple document formats and answers questions using a sophisticated multi-agent architecture.

## Architecture

1. **CoordinatorAgent**
  - Orchestrates the entire workflow
  - Manages trace IDs and workflow states
  - Handles error coordination
2. **IngestionAgent**
  - Processes multiple document formats (PDF, PPTX, CSV, DOCX, TXT, MD)
  - Chunks text using RecursiveCharacterTextSplitter
  - Handles document parsing and preprocessing
3. **RetrievalAgent**
  - Creates and manages FAISS vector embeddings
  - Performs similarity search for context retrieval
  - Uses Google Generative AI embeddings
4. **LLMResponseAgent**
  - Generates responses using Google's Gemini model
  - Uses LangChain QA chain for structured responses
  - Maintains conversation context

## Technologies Used:
- **Streamlit**:  For building the web interface
- **Document Processing**: PyPDF2, python-pptx, pandas, python-docx
- **LangChain**: For managing text processing and embeddings
- **Google Generative AI (Gemini)**: For embeddings and AI-driven responses
- **FAISS**: For storing and searching text embeddings
- **dotenv**: For managing API keys securely
- **Async Processing**: asyncio

## Installation & Setup:

1. Clone the Repository
   ```bash
    git clone https://github.com/yourusername/agentic-rag-chatbot.git
    cd agentic-rag-chatbot
    ```
2. Create and Activate Virtual Environment
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install the required Python libraries.
   ```bash
    pip install -r requirements.txt
    ```
4. Set Up API Key Create a .env file in the root directory and add your Google API key:
   ```bash
    GOOGLE_API_KEY=your_google_api_key_here
    ```
5. Run the application
   ```bash
    streamlit run app.py
    ```

## How to Use:

1. **Start the application**
   ```bash
    streamlit run app.py
    ```
2. **Upload Documents**
  - Use the sidebar to upload multiple documents
  - Supported formats: PDF, PPTX, CSV, DOCX, TXT, MD
  - Click "Process" for each document
3. **Ask Questions**
  - Type questions in the chat interface
  - The system will retrieve relevant context and generate responses
  - View sources and system monitoring information
## UI Features
- Multi-column Layout: Organized sidebar and main chat area
- Real-time Processing: Live status updates during document processing
- Chat History: Persistent conversation history
- Source Attribution: Shows which documents contributed to answers
- System Monitor: MCP message tracking and agent status
