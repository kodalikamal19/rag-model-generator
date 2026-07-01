import os
import glob
import sys
import warnings

# Suppress deprecation warnings from LangChain components
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import config

# Import LangChain components in a version-safe manner
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

def get_embeddings_model():
    """Initializes and returns the configured embedding model."""
    if config.EMBEDDING_MODEL_TYPE == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Please configure it first.")
        from langchain_openai import OpenAIEmbeddings
        print("Initializing OpenAI Embeddings...")
        return OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    elif config.EMBEDDING_MODEL_TYPE == "gemini":
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set. Please configure it first.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        print("Initializing Google Gemini Embeddings...")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=config.GOOGLE_API_KEY)
    else:
        # Default fallback to HuggingFace
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            from langchain.embeddings import HuggingFaceEmbeddings
        print("Initializing HuggingFace Embeddings ('all-MiniLM-L6-v2')...")
        print("(This is running locally on CPU, no API keys required and free of charge)")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_documents(directory: str):
    """Loads all PDF, TXT, and DOCX files from the specified folder."""
    documents = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Documents directory '{directory}' does not exist.")

    # Walk directory to find files
    pdf_files = glob.glob(os.path.join(directory, "**/*.pdf"), recursive=True)
    txt_files = glob.glob(os.path.join(directory, "**/*.txt"), recursive=True)
    docx_files = glob.glob(os.path.join(directory, "**/*.docx"), recursive=True)
    
    total_files = len(pdf_files) + len(txt_files) + len(docx_files)
    if total_files == 0:
        print(f"No documents (.pdf, .txt, .docx) found in directory '{directory}'.")
        return [], 0, 0, 0

    print(f"Found {total_files} file(s) to process:")
    print(f" - PDF: {len(pdf_files)} files")
    print(f" - TXT: {len(txt_files)} files")
    print(f" - DOCX: {len(docx_files)} files")

    # Load TXT files
    for file_path in txt_files:
        try:
            print(f"Loading text document: {os.path.basename(file_path)}")
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Load PDF files
    for file_path in pdf_files:
        try:
            print(f"Loading PDF document: {os.path.basename(file_path)}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Load DOCX files
    for file_path in docx_files:
        try:
            print(f"Loading Word document: {os.path.basename(file_path)}")
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents, len(pdf_files), len(txt_files), len(docx_files)

def ingest_documents_pipeline() -> dict:
    """Executes the complete document ingestion pipeline and returns statistics."""
    print("=" * 60)
    print("              *** DOCUMENT INGESTION PROCESS ***")
    print("=" * 60)
    
    print(f"Document Source Directory: {config.DOCUMENTS_DIR}")
    print(f"Vector Database Target:   {config.VECTORSTORE_DIR}")
    print(f"Splitter Settings:        Chunk Size = {config.CHUNK_SIZE}, Overlap = {config.CHUNK_OVERLAP}")
    
    # 1. Load documents
    raw_docs, pdf_count, txt_count, docx_count = load_documents(config.DOCUMENTS_DIR)
    if not raw_docs:
        return {
            "status": "warning",
            "message": "No documents found to ingest.",
            "chunks_created": 0,
            "files_processed": 0
        }

    # 2. Split documents
    print("\nSplitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Created {len(chunks)} chunks from {len(raw_docs)} loaded pages/documents.")

    # 3. Get Embeddings Model
    embeddings = get_embeddings_model()

    # 4. Generate Embeddings & Save Vector Database
    print(f"\nBuilding {config.VECTOR_DB_TYPE.upper()} Vector Store...")
    
    if config.VECTOR_DB_TYPE == "chroma":
        try:
            from langchain_chroma import Chroma
        except ImportError:
            try:
                from langchain_community.vectorstores import Chroma
            except ImportError:
                from langchain.vectorstores import Chroma
            
        print(f"Saving vector database to directory: {config.VECTORSTORE_DIR} ...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=config.VECTORSTORE_DIR
        )
        print("[OK] Database persisted successfully.")
        
    elif config.VECTOR_DB_TYPE == "faiss":
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            from langchain.vectorstores import FAISS
            
        print(f"Building FAISS index...")
        db = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        print(f"Saving FAISS index locally to: {config.VECTORSTORE_DIR} ...")
        db.save_local(config.VECTORSTORE_DIR)
        print("[OK] Index saved successfully.")
        
    else:
        raise ValueError(f"Unknown vector database type '{config.VECTOR_DB_TYPE}'")

    print("\n" + "=" * 60)
    print("         *** DOCUMENT INGESTION COMPLETE ***")
    print("=" * 60)
    
    return {
        "status": "success",
        "message": f"Successfully ingested {pdf_count + txt_count + docx_count} file(s).",
        "chunks_created": len(chunks),
        "files_processed": pdf_count + txt_count + docx_count,
        "details": {
            "pdfs": pdf_count,
            "txts": txt_count,
            "docxs": docx_count
        }
    }

def main():
    try:
        result = ingest_documents_pipeline()
        print(result["message"])
        if result["status"] == "success":
            print(f"Created {result['chunks_created']} text chunks in {config.VECTOR_DB_TYPE.upper()} vector store.")
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
