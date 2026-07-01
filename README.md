# AI-Powered RAG Builder 🤖

A simple, modular, and fully functional RAG (Retrieval-Augmented Generation) pipeline generator. It automatically selects optimal parameters (chunk size, overlap, embedding models, vector database, and retrieval modes) based on a natural language description of your project requirements, builds a pipeline, ingests documents, and deploys a FastAPI server exposing query endpoints and a beautiful web dashboard.

---

## Project Structure

```text
project/
│
├── app.py              # FastAPI server implementing the RAG endpoint & serving web UI
├── config.py           # Exports project configurations loaded from config.json
├── rag_builder.py      # Interactive CLI requirement analyzer and builder
├── ingest.py           # Parses documents, generates embeddings, and saves vector store
├── index.html          # Web dashboard interface (single page HTML/CSS/JS)
├── requirements.txt    # Python package dependencies
├── Dockerfile          # Container setup for easy production deployment
├── documents/          # User source documents (.txt, .pdf, .docx)
├── vectorstore/        # Saved vector database index
└── README.md           # This instruction manual
```

---

## Getting Started: Step-by-Step

Follow these instructions to configure, build, and run the RAG pipeline from scratch.

### Step 1: Create a Virtual Environment

Open a terminal in the project directory and create a virtual environment to isolate the project dependencies:

```bash
# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Web Server and Dashboard

Start the FastAPI application server using Uvicorn:

```bash
uvicorn app:app --reload
```

The application runs locally at: **`http://127.0.0.1:8000`**

Open this URL in your web browser. You will see a beautiful dark-themed dashboard.

---

## Operating via the Web Dashboard 🖥️

Inside the browser, you can control the entire RAG pipeline visually:

1. **Configure Parameters**:
   - Write your requirement description in the text box (e.g. *"I need highly accurate answers with citations. Use OpenAI."*).
   - Select your LLM Provider.
   - Enter your API key (if using OpenAI/Gemini; this writes it to a local `.env` file).
   - Click **Analyze & Build Configuration**.
2. **Ingest Documents**:
   - The files in your `documents/` folder will automatically list on the dashboard.
   - Click **Ingest & Vectorize Documents** to split, embed, and index them.
3. **Query/Chat**:
   - Type your question in the chat pane and press **Ask**. The system retrieves context and prints the AI response along with source reference files.

---

## API Endpoints Reference 🔌

The FastAPI backend exposes the query interfaces in two formats:

### 1. GET Method (URL Parameter)
Ideal for quick URL queries, simple scripts, or embeds where parameters are passed in the URL.

* **Endpoint**: `GET /query`
* **Query Parameter**: `question` (string)
* **Example URL**: `http://127.0.0.1:8000/query?question=What+is+the+remote+work+policy?`
* **Response**:
  ```json
  {
    "answer": "Acme Corp allows employees to work remotely up to three days per week...",
    "sources": ["sample_faq.txt"]
  }
  ```

### 2. POST Method (JSON Body)
Ideal for standard programmatic requests, web applications, or SDK integrations.

* **Endpoint**: `POST /query`
* **Request Body**:
  ```json
  {
    "question": "Is there an internet stipend?"
  }
  ```
* **Response**:
  ```json
  {
    "answer": "Yes, remote and hybrid employees receive an internet allowance of $50 per month.",
    "sources": ["sample_faq.txt"]
  }
  ```

---

## Deployment Guide 🚀

To make your RAG API accessible to anyone on the web, follow these hosting options:

### A. Pre-built Database Recommendation (Stateless Cloud)
If you deploy to a free-tier hosting provider (which are usually stateless and reset their local storage on container restarts), new documents ingested through the UI will be lost on restart.
* **Best Practice**:
  1. Add your documents to your local `documents/` folder.
  2. Run `python rag_builder.py` and `python ingest.py` locally.
  3. Commit the created `config.json` and the `vectorstore/` folder containing your pre-compiled index database directly into your Git repository.
  4. Push it to your deployment platform. This packages your active vector database directly inside the deployment build!

### B. Deployment to Render.com (Free Tier)
1. Push your project to **GitHub**.
2. Create a free account at [Render.com](https://render.com).
3. Click **New +** ➔ **Web Service**.
4. Connect your GitHub repository.
5. In the settings:
   - **Environment**: Select **Docker** (Render will automatically pick up the local `Dockerfile`).
   - **Branch**: Select `main`.
6. Click **Advanced** and add your environment variables:
   - `OPENAI_API_KEY`: `your-actual-api-key-here` (if using OpenAI)
   - `GOOGLE_API_KEY`: `your-actual-api-key-here` (if using Gemini)
7. Click **Deploy Web Service**. Render will build and host your RAG server, exposing a public URL like `https://rag-model-generator.onrender.com`.

### C. Deployment to Hugging Face Spaces (Free CPU Tier)
1. Create a free account at [Hugging Face](https://huggingface.co).
2. Click your profile picture ➔ **New Space**.
3. In settings:
   - **Space SDK**: Select **Docker**.
   - **Template**: Select **Blank**.
   - **License**: Select `apache-2.0` (or preference).
4. Clone the space Git repository locally, copy your project files into it, commit, and push.
5. Add your API keys under Space **Settings** ➔ **Variables and secrets**.
6. Hugging Face will build the container and deploy your FastAPI server at the public Space URL!

---

## Troubleshooting & Common Errors

### 1. `ModuleNotFoundError: No module named 'langchain.chains'`
* **Fix**: This happens if your python environment relies on a custom `langchain` layout. The updated builder code includes automated compatibility fallbacks to detect both standard `langchain` and `langchain_classic` packages automatically. Ensure you run the server using `venv\Scripts\python -m uvicorn app:app --reload` to execute inside the correct environment.

### 2. `Warning: Vectorstore directory is empty`
* **Fix**: You started the FastAPI app before running the ingestion pipeline. Place documents in `./documents` and run `python ingest.py` or hit **Ingest Documents** on the web dashboard.

### 3. API Key Missing / Unauthorized Errors
* **Fix**: Make sure you have a valid API key set in your `.env` file or cloud dashboard variables.

### 4. Running Local/Ollama LLMs
* If you chose Local / Ollama, you must have Ollama running on your machine.
* **Fix**: Start the Ollama application, then pull and run the model in your command prompt:
  ```bash
  ollama run llama3
  ```

### 5. `sqlite3` version compatibility issues (ChromaDB)
* ChromaDB requires a newer version of SQLite. If you get a sqlite3 version mismatch error on Windows:
  * **Fix**: Switch your vector DB selection to **FAISS** in the builder dashboard or update `config.json`'s `"vector_db_type": "faiss"` and re-run ingestion. FAISS works in memory/binary format and does not rely on sqlite.
