import os
import sys
import json
import warnings
import importlib
from typing import List, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Suppress deprecation and user warnings from third-party libraries (e.g. Chroma, LangChain)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import config and helper to check environment
import config

# Import LangChain BaseChatModel to implement fallback LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, AIMessage

# Define Fallback Mock LLM in case credentials or local LLM setup is missing
class FallbackMockChatModel(BaseChatModel):
    error_message: str = "LLM configuration error."
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Returns a helpful warning message instead of crashing
        warning_text = (
            f"[WARNING] RAG Pipeline Fallback: The configured LLM could not be started.\n"
            f"Reason: {self.error_message}\n\n"
            f"Please verify your credentials and ensure your target server is running."
        )
        message = AIMessage(content=warning_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
        
    @property
    def _llm_type(self) -> str:
        return "fallback_mock"

# Global RAG variables
db = None
retriever = None
rag_chain = None

def get_embeddings_model():
    """Retrieves embedding model based on settings."""
    if config.EMBEDDING_MODEL_TYPE == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    else:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            from langchain.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def init_llm():
    """Initializes LLM or returns the fallback mock LLM on error."""
    try:
        if config.LLM_TYPE == "openai":
            if not config.OPENAI_API_KEY:
                return FallbackMockChatModel(
                    error_message="OPENAI_API_KEY environment variable is not set. Run 'python rag_builder.py' to set it."
                )
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.LLM_MODEL_NAME, 
                openai_api_key=config.OPENAI_API_KEY, 
                temperature=0
            )
            
        elif config.LLM_TYPE == "gemini":
            if not config.GOOGLE_API_KEY:
                return FallbackMockChatModel(
                    error_message="GOOGLE_API_KEY environment variable is not set. Run 'python rag_builder.py' to set it."
                )
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=config.LLM_MODEL_NAME, 
                google_api_key=config.GOOGLE_API_KEY, 
                temperature=0
            )
            
        elif config.LLM_TYPE == "local":
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    model=config.LLM_MODEL_NAME, 
                    base_url=config.OLLAMA_BASE_URL, 
                    temperature=0
                )
            except Exception as ex:
                return FallbackMockChatModel(
                    error_message=f"Could not connect to local Ollama server at {config.OLLAMA_BASE_URL}. "
                                  f"Make sure Ollama is installed, running, and the '{config.LLM_MODEL_NAME}' model is pulled. Error details: {ex}"
                )
        else:
            return FallbackMockChatModel(error_message=f"Unsupported LLM provider: {config.LLM_TYPE}")
            
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return FallbackMockChatModel(error_message=f"Unexpected error configuring LLM: {e}")

def init_rag_pipeline():
    """Initializes vector store and constructs the LangChain QA chain with version fallbacks."""
    global db, retriever, rag_chain
    
    # Verify that the vector database exists
    if not os.path.exists(config.VECTORSTORE_DIR) or not os.listdir(config.VECTORSTORE_DIR):
        print(f"Warning: Vectorstore directory '{config.VECTORSTORE_DIR}' is empty or does not exist.")
        return False
        
    print(f"Loading vector database ({config.VECTOR_DB_TYPE.upper()}) from: {config.VECTORSTORE_DIR}...")
    embeddings = get_embeddings_model()
    
    # Load database
    if config.VECTOR_DB_TYPE == "chroma":
        try:
            from langchain_chroma import Chroma
        except ImportError:
            try:
                from langchain_community.vectorstores import Chroma
            except ImportError:
                from langchain.vectorstores import Chroma
        db = Chroma(persist_directory=config.VECTORSTORE_DIR, embedding_function=embeddings)
        
    elif config.VECTOR_DB_TYPE == "faiss":
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            from langchain.vectorstores import FAISS
        db = FAISS.load_local(config.VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Error: Unknown vector DB type '{config.VECTOR_DB_TYPE}'")
        return False

    # Configure retriever based on settings
    if config.RETRIEVER_TYPE == "mmr":
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
    else:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
    # Connect LLM
    llm = init_llm()
    
    # Version-safe import of Langchain retrieval chains
    try:
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
    except ImportError:
        try:
            # Fallback to langchain_classic if standard chains fails
            from langchain_classic.chains import create_retrieval_chain
            from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        except ImportError:
            print("Critical: Could not import create_retrieval_chain or create_stuff_documents_chain.")
            return False

    from langchain_core.prompts import ChatPromptTemplate
    
    # Context-aware instructions
    system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved context "
        "to answer the question. If you do not know the answer based on the context, "
        "state clearly that you do not know. Keep your answer brief and factual.\n\n"
        "Context:\n{context}"
    )
    
    # Request source citation references if enabled
    if config.CITATIONS_ENABLED:
        system_prompt += (
            "\n\nAt the end of your answer, include citations pointing to the files "
            "used to construct your answer (e.g. [filename.pdf] or [filename.txt])."
        )
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    print("[OK] RAG pipeline successfully initialized.")
    return True

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try initialization at startup
    init_rag_pipeline()
    yield
    # Cleanup if needed

# Initialize FastAPI application
app = FastAPI(
    title="AI-Powered RAG Builder API",
    description="Automatically built and deployed RAG search API.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas for configuration
class ConfigureRequest(BaseModel):
    description: str = Field(..., example="I need highly accurate answers with citations. Speed is not important. Use OpenAI.")
    llm_choice: str = Field("openai", example="openai", description="Choose 'openai', 'gemini', or 'local'")
    api_key: Optional[str] = Field(None, example="sk-...")
    documents_dir: str = Field("./documents", example="./documents")

class QueryRequest(BaseModel):
    question: str = Field(..., example="What is the remote work policy at Acme Corp?")

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# Serve Frontend HTML
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(config.BASE_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<html><body><h1>HTML file index.html not found. Please create it in the workspace directory.</h1></body></html>"

# API Endpoints
@app.post("/api/configure")
async def configure_rag(request: ConfigureRequest):
    """Parses requirements dynamically, updates config.json, updates .env, and reloads the pipeline."""
    try:
        from rag_builder import parse_requirements
        
        # Map LLM choice string to numeric input expected by parse_requirements
        llm_map = {
            "openai": "1",
            "gemini": "2",
            "local": "3"
        }
        numeric_choice = llm_map.get(request.llm_choice.lower(), "1")
        
        # Run rule-based requirements analysis
        parsed_config = parse_requirements(request.description, numeric_choice)
        parsed_config["documents_dir"] = request.documents_dir
        parsed_config["vectorstore_dir"] = "./vectorstore"
        
        # Save config.json
        with open(config.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(parsed_config, f, indent=4)
            
        # Update .env if a key was provided
        if request.api_key:
            env_path = os.path.join(config.BASE_DIR, ".env")
            env_line = ""
            if parsed_config["llm_type"] == "openai":
                env_line = f"OPENAI_API_KEY={request.api_key}"
            elif parsed_config["llm_type"] == "gemini":
                env_line = f"GOOGLE_API_KEY={request.api_key}"
                
            if env_line:
                mode = "a" if os.path.exists(env_path) else "w"
                with open(env_path, mode, encoding="utf-8") as f:
                    if mode == "a":
                        f.write("\n")
                    f.write(env_line + "\n")
                    
        # Dynamically reload configuration settings in memory
        importlib.reload(config)
        
        # Reset and reload the pipeline
        global rag_chain, db, retriever
        db = None
        retriever = None
        rag_chain = None
        init_rag_pipeline()
        
        return {
            "status": "success",
            "message": "Configuration updated successfully and RAG pipeline reinitialized.",
            "config": parsed_config
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while updating the configuration: {str(e)}"
        )

@app.get("/api/documents")
async def get_documents():
    """Lists files inside the configured documents directory."""
    docs_dir = os.path.abspath(config.DOCUMENTS_DIR)
    if not os.path.exists(docs_dir):
        return {"documents": [], "path": docs_dir}
    
    files = []
    for f in os.listdir(docs_dir):
        if f.endswith((".pdf", ".txt", ".docx")):
            files.append(f)
            
    return {"documents": files, "path": docs_dir}

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Saves uploaded files (PDF, TXT, DOCX) to the documents directory."""
    docs_dir = os.path.abspath(config.DOCUMENTS_DIR)
    os.makedirs(docs_dir, exist_ok=True)
    
    saved_files = []
    errors = []
    
    for file in files:
        if not file.filename.endswith((".pdf", ".txt", ".docx")):
            errors.append(f"Rejected {file.filename}: Unsupported file format (only PDF, TXT, DOCX allowed).")
            continue
        try:
            file_path = os.path.join(docs_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(file.filename)
        except Exception as e:
            errors.append(f"Failed to save {file.filename}: {str(e)}")
            
    if len(saved_files) == 0 and errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))
        
    return {
        "status": "success",
        "message": f"Successfully uploaded {len(saved_files)} file(s).",
        "uploaded_files": saved_files,
        "errors": errors
    }

@app.post("/api/ingest")
async def run_api_ingest():
    """Runs the document ingestion process dynamically and reloads the vector store."""
    try:
        from ingest import ingest_documents_pipeline
        
        # Execute ingestion pipeline
        result = ingest_documents_pipeline()
        
        # Reload pipeline with the newly indexed data
        if result["status"] == "success":
            global rag_chain, db, retriever
            db = None
            retriever = None
            rag_chain = None
            init_rag_pipeline()
            
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during ingestion: {str(e)}"
        )

async def execute_query_logic(question: str) -> QueryResponse:
    global db, retriever, rag_chain
    
    # Check if pipeline needs initialization (e.g. if files were ingested after server startup)
    if not rag_chain:
        initialized = init_rag_pipeline()
        if not initialized:
            raise HTTPException(
                status_code=400,
                detail="The vector database is uninitialized or empty. Please run document ingestion first: 'python ingest.py'"
            )
            
    try:
        # Run query
        response = rag_chain.invoke({"input": question})
        
        answer = response.get("answer", "")
        context_docs = response.get("context", [])
        
        # Extract sources from documents
        sources = []
        for doc in context_docs:
            source_path = doc.metadata.get("source", "unknown")
            source_name = os.path.basename(source_path)
            page = doc.metadata.get("page")
            
            if page is not None:
                sources.append(f"{source_name} (page {page + 1})")
            else:
                sources.append(source_name)
                
        # Deduplicate sources
        unique_sources = sorted(list(set(sources)))
        
        return QueryResponse(answer=answer, sources=unique_sources)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while running the query: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Processes a user question against the vectorstore and generates an LLM answer via JSON body."""
    return await execute_query_logic(request.question)

@app.get("/query", response_model=QueryResponse)
async def query_rag_get(question: str = Field(..., description="The query question to search.")):
    """Processes a user question against the vectorstore via URL query parameter: /query?question=Your+Question"""
    return await execute_query_logic(question)

@app.get("/health")
async def health_check():
    """Returns the status of the RAG application and its configuration."""
    pipeline_loaded = rag_chain is not None
    return {
        "status": "online" if pipeline_loaded else "waiting_for_ingestion",
        "pipeline_initialized": pipeline_loaded,
        "config": {
            "llm_type": config.LLM_TYPE,
            "llm_model": config.LLM_MODEL_NAME,
            "vector_db": config.VECTOR_DB_TYPE,
            "embeddings": config.EMBEDDING_MODEL_TYPE,
            "citations_enabled": config.CITATIONS_ENABLED
        }
    }
