import os
import json
import sys

def parse_requirements(description: str, llm_choice: str):
    """Analyzes the user's natural language requirements and selects RAG parameters."""
    desc_lower = description.lower()
    
    # 1. Determine Chunk Size & Overlap based on speed/accuracy requirements
    # High Accuracy: small chunks with good overlap (retains local context)
    # High Speed/Performance: larger chunks, less overlap (fewer chunks to retrieve & process)
    accuracy_keywords = ["accuracy", "accurate", "precise", "precision", "detail", "citation", "reference", "quality", "deep"]
    speed_keywords = ["speed", "fast", "quick", "latency", "real-time", "performance", "instant"]
    
    # Simple check for speed/accuracy negation
    speed_negations = ["speed is not", "speed not", "no speed", "unimportant", "not important"]
    is_speed_negated = any(neg in desc_lower for neg in speed_negations)
    
    is_accuracy = any(kw in desc_lower for kw in accuracy_keywords)
    is_speed = any(kw in desc_lower for kw in speed_keywords) and not is_speed_negated
    
    if is_accuracy and not is_speed:
        chunk_size = 400
        chunk_overlap = 100
        retriever_type = "mmr"  # MMR retriever ensures diverse results for accurate synthesis
    elif is_speed and not is_accuracy:
        chunk_size = 800
        chunk_overlap = 50
        retriever_type = "similarity"
    else:
        # Default balanced settings
        chunk_size = 500
        chunk_overlap = 100
        retriever_type = "similarity"
        
    # 2. Determine Vector Database based on scale
    large_scale_keywords = ["large", "huge", "scale", "million", "thousands", "many", "1000", "faiss", "big"]
    is_large_scale = any(kw in desc_lower for kw in large_scale_keywords)
    
    if is_large_scale:
        vector_db_type = "faiss"
    else:
        vector_db_type = "chroma"  # Chroma is great for persistent small/medium projects
        
    # 3. Determine Citation requirements
    citation_keywords = ["citation", "cite", "reference", "source", "sources", "footnote"]
    citations_enabled = any(kw in desc_lower for kw in citation_keywords)
    
    # 4. LLM settings
    llm_map = {
        "1": "openai",
        "2": "gemini",
        "3": "local"
    }
    llm_type = llm_map.get(llm_choice.strip(), "openai")
    
    # Map default model names
    if llm_type == "openai":
        llm_model_name = "gpt-4o-mini"
        embedding_model_type = "openai"  # OpenAI LLM can default to OpenAI embeddings if key is provided
    elif llm_type == "gemini":
        llm_model_name = "gemini-1.5-flash"
        embedding_model_type = "huggingface"  # Free HuggingFace embeddings
    else:
        llm_model_name = "llama3"
        embedding_model_type = "huggingface"  # Free HuggingFace embeddings
        
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "vector_db_type": vector_db_type,
        "embedding_model_type": embedding_model_type,
        "llm_type": llm_type,
        "llm_model_name": llm_model_name,
        "retriever_type": retriever_type,
        "citations_enabled": citations_enabled
    }

def print_banner():
    print("=" * 60)
    print("        *** WELCOME TO THE AI-POWERED RAG BUILDER ***")
    print("=" * 60)
    print("Describe your requirements in plain English, and we will")
    print("automatically analyze, configure, and prepare your RAG API.")
    print("=" * 60)

def main():
    print_banner()
    
    # Prompt for user requirements
    print("\n[Step 1] Describe your RAG requirements:")
    default_desc = "I have around 100 PDF files. I need highly accurate answers with citations. Speed is not important. Use OpenAI."
    print(f"Example: {default_desc}")
    description = input("Your description: ").strip()
    if not description:
        description = default_desc
        print(f"Using default: '{description}'")
        
    # Prompt for LLM Choice
    print("\n[Step 2] Select your LLM Provider:")
    print("1. OpenAI (Requires API Key, default)")
    print("2. Gemini (Requires API Key)")
    print("3. Local / Ollama (Requires Ollama running locally)")
    llm_choice = input("Select option [1-3]: ").strip()
    if llm_choice not in ["1", "2", "3"]:
        llm_choice = "1"
        
    # Prompt for documents folder
    print("\n[Step 3] Configure Documents Folder:")
    documents_dir = input("Enter path to documents folder [default: ./documents]: ").strip()
    if not documents_dir:
        documents_dir = "./documents"
        
    # Process options and perform rule-based analysis
    config = parse_requirements(description, llm_choice)
    config["documents_dir"] = documents_dir
    config["vectorstore_dir"] = "./vectorstore"
    
    # Handle API Keys
    api_key = ""
    env_lines = []
    
    if config["llm_type"] == "openai":
        print("\n[Step 4] OpenAI API Key:")
        api_key = input("Enter your OpenAI API key (leave blank if already set in environment): ").strip()
        if api_key:
            env_lines.append(f"OPENAI_API_KEY={api_key}")
        # If using OpenAI, offer choice of embeddings
        print("\nUse OpenAI Embeddings or free HuggingFace Embeddings?")
        print("1. HuggingFace Embeddings (Free, local, recommended) [default]")
        print("2. OpenAI Embeddings (Paid, requires API credits)")
        embed_choice = input("Select embedding option [1-2]: ").strip()
        if embed_choice != "2":
            config["embedding_model_type"] = "huggingface"
            
    elif config["llm_type"] == "gemini":
        print("\n[Step 4] Google Gemini API Key:")
        api_key = input("Enter your GOOGLE_API_KEY (leave blank if already set in environment): ").strip()
        if api_key:
            env_lines.append(f"GOOGLE_API_KEY={api_key}")
        config["embedding_model_type"] = "huggingface"
        
    elif config["llm_type"] == "local":
        print("\n[Step 4] Local Setup:")
        print("Make sure Ollama is installed and running (`ollama run llama3`).")
        config["embedding_model_type"] = "huggingface"
        
    # Write to .env file if new keys were input
    if env_lines:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        # Append or write
        mode = "a" if os.path.exists(env_path) else "w"
        with open(env_path, mode, encoding="utf-8") as f:
            if mode == "a":
                f.write("\n")
            f.write("\n".join(env_lines) + "\n")
        print("[OK] API key saved to .env file.")
        
    # Print the analysis results
    print("\n" + "=" * 40)
    print("         ANALYSIS RESULTS")
    print("=" * 40)
    print(f"LLM Provider:     {config['llm_type'].upper()} ({config['llm_model_name']})")
    print(f"Embedding Model:  {config['embedding_model_type'].upper()}")
    print(f"Vector Database:  {config['vector_db_type'].upper()}")
    print(f"Chunk Size:       {config['chunk_size']} characters")
    print(f"Chunk Overlap:    {config['chunk_overlap']} characters")
    print(f"Retriever Type:   {config['retriever_type'].upper()}")
    print(f"Citations Needed: {'YES' if config['citations_enabled'] else 'NO'}")
    print("=" * 40)
    
    # Save the config.json
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"[OK] Configuration saved to config.json")
    
    # Create directories
    docs_path = os.path.abspath(os.path.join(base_dir, documents_dir))
    os.makedirs(docs_path, exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(base_dir, config["vectorstore_dir"])), exist_ok=True)
    print(f"[OK] Created directory: {docs_path}")
    
    # Generate Sample FAQ document
    sample_file_path = os.path.join(docs_path, "sample_faq.txt")
    if not os.path.exists(sample_file_path):
        sample_content = """Frequently Asked Questions - Acme Corp

Q: What is the remote work policy at Acme Corp?
A: Acme Corp allows employees to work remotely up to three days per week, subject to approval from their department manager.

Q: Is there an internet stipend?
A: Yes, remote and hybrid employees receive a high-speed internet allowance of $50 per month to support their home workspace setup.

Q: What are the core working hours?
A: The core hours when all employees should be online and available are from 10:00 AM to 3:00 PM EST.

Q: Who is the CEO of Acme Corp?
A: The CEO of Acme Corp is Dr. Jane Foster, appointed in January 2024.
"""
        with open(sample_file_path, "w", encoding="utf-8") as f:
            f.write(sample_content)
        print(f"[OK] Generated sample document at: {sample_file_path}")
        
    print("\nNext steps:")
    print("1. Install requirements:  pip install -r requirements.txt")
    print("2. (Optional) Place more documents in: " + documents_dir)
    print("3. Ingest documents:      python ingest.py")
    print("4. Start FastAPI server:  uvicorn app:app --reload")
    print("=" * 60)

if __name__ == "__main__":
    main()
