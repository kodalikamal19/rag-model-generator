import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# Default settings used as a fallback if config.json is not yet generated
DEFAULT_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 100,
    "vector_db_type": "chroma",
    "embedding_model_type": "huggingface",
    "llm_type": "openai",
    "llm_model_name": "gpt-4o-mini",
    "retriever_type": "similarity",
    "citations_enabled": False,
    "documents_dir": os.path.join(BASE_DIR, "documents"),
    "vectorstore_dir": os.path.join(BASE_DIR, "vectorstore")
}

def load_config():
    """Loads configuration parameters from config.json if it exists."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                # Resolve relative paths relative to BASE_DIR if they start with ./
                for key in ["documents_dir", "vectorstore_dir"]:
                    if key in user_config and user_config[key].startswith("./"):
                        user_config[key] = os.path.abspath(os.path.join(BASE_DIR, user_config[key].replace("./", "", 1)))
                
                # Merge user config with default configuration
                return {**DEFAULT_CONFIG, **user_config}
        except Exception as e:
            print(f"Warning: Failed to load config.json ({e}). Using defaults.")
    return DEFAULT_CONFIG

# Load configuration settings
config_data = load_config()

# Export config settings as uppercase module constants
CHUNK_SIZE = int(config_data.get("chunk_size", 500))
CHUNK_OVERLAP = int(config_data.get("chunk_overlap", 100))
VECTOR_DB_TYPE = config_data.get("vector_db_type", "chroma").lower()
EMBEDDING_MODEL_TYPE = config_data.get("embedding_model_type", "huggingface").lower()
LLM_TYPE = config_data.get("llm_type", "openai").lower()
LLM_MODEL_NAME = config_data.get("llm_model_name", "gpt-4o-mini")
RETRIEVER_TYPE = config_data.get("retriever_type", "similarity").lower()
CITATIONS_ENABLED = bool(config_data.get("citations_enabled", False))
DOCUMENTS_DIR = config_data.get("documents_dir", os.path.join(BASE_DIR, "documents"))
VECTORSTORE_DIR = config_data.get("vectorstore_dir", os.path.join(BASE_DIR, "vectorstore"))

# API Keys and External Service URLs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Standardize default LLM models based on provider selection
if LLM_TYPE == "openai" and LLM_MODEL_NAME == "gpt-4o-mini":
    # User might specify different models, default is gpt-4o-mini
    pass
elif LLM_TYPE == "gemini" and LLM_MODEL_NAME == "gemini-1.5-flash":
    pass
elif LLM_TYPE == "local" and LLM_MODEL_NAME == "llama3":
    pass
