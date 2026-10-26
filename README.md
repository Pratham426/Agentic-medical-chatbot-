# Agentic-medical-chatbot-


An educational medical information assistant built with FastAPI, LangGraph, Groq, Pinecone, Sentence Transformers, and optional Tavily web search. It supports medical question answering, retrieval-augmented generation (RAG), image observations, and conversation threads.

## Models and Services

- **Chat model:** `openai/gpt-oss-20b` via Groq
- **Vision model:** `qwen/qwen3.8-27b` via Groq, provided the model is available and supports image input on your Groq account
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` running locally
- **Vector database:** Pinecone (`medibot-vectors`)
- **Retrieval namespace:** `medical`
- **Web search:** Tavily (optional)
- **Speech transcription:** Groq Whisper, if supported by the configured API
- **Text-to-speech:** Requires a separate TTS provider

## Setup

### 1. Create and activate a virtual environment

**Windows — Git Bash:**

```bash
python -m venv Medibot
source Medibot/Scripts/activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Install the packages listed in your project's `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

If you do not have a requirements file, create one containing the dependencies used by your application, including FastAPI, Uvicorn, LangGraph, LangChain Groq, Pinecone, Sentence Transformers, and any other packages imported by your project.

### 3. Configure environment variables

Create a `.env` file in the project root:


GROQ_CHAT_MODEL=openai/gpt-oss-20b
GROQ_VISION_MODEL=qwen/qwen3.8-27b
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

API_TOKEN=hello
ALLOWED_ORIGINS=http://localhost:8000,https://127.0.0.1:8000

```

Replace the placeholder API keys with your own credentials. Tavily is optional. Ensure `app/config.py` reads these exact environment variable names and maps them to the settings used by your application.

**Security:** `API_TOKEN=hello` is suitable only for local testing. Use a strong secret for any shared environment, and never commit `.env` to Git.

### 4. Start the application

From the project root, run:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open the application:

- Web interface: http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/api/health

The health endpoint checks whether required configuration values are present; it does not guarantee that every external service is reachable.

## Knowledge Base and Embeddings

The project uses `sentence-transformers/all-MiniLM-L6-v2` to generate embeddings locally. Its embedding dimension is **384**, so it does not require paid OpenAI embedding API calls.

The Pinecone index `medibot-vectors` is configured for 384-dimensional dense vectors. Medical documents are stored in the `medical` namespace.

The existing index uses Pinecone's document-schema search API. Retrieval code must use the appropriate document search method and the `medical` namespace.

If your medical documents are already ingested, you do not need to ingest them again merely to start the application. Run your ingestion script only when the knowledge base needs to be created or updated, and ensure it writes records in the same namespace and schema expected by your retrieval code.

## Project Structure

- `app/config.py` — environment variables and application settings
- `app/knowledge.py` — Groq chat client, local embeddings, and Pinecone retrieval
- `app/agent.py` — LangGraph workflow for retrieval, routing, optional web search, and answer generation
- `app/main.py` — FastAPI application, chat endpoint, image observations, transcription, and speech endpoint
- `ingest.py` — document processing and knowledge-base ingestion
- `templates/index.html` — web interface
- `static/style.css` — interface styles
- `static/app.js` — browser-side interactions
- `data/` — source medical reference documents

## Troubleshooting

- **Application won't start:** Activate the correct virtual environment, install dependencies, and check the terminal traceback.
- **401 Unauthorized:** If `API_TOKEN` is configured, send `Authorization: Bearer your_token` with API requests. Ensure the UI sends the same token.
- **502 AI service error:** Check the backend terminal for the underlying exception. Verify Groq and Pinecone credentials, network connectivity, model availability, and Pinecone schema.
- **Pinecone retrieval errors:** Confirm the index is `medibot-vectors`, the namespace is `medical`, the dense vector field is `chunk_text`, and the embedding dimension is 384.
- **No knowledge-base results:** Verify that the `medical` namespace contains records and that stored document fields include the actual medical text. Check ingestion and retrieval schemas before re-ingesting.
- **Vision errors:** Confirm the configured Groq model supports image input and that the image is valid JPEG, PNG, or WebP within the application's size limit.
- **Transcription errors:** Confirm the transcription model is supported by the installed Groq SDK and your account.
- **Text-to-speech errors:** Groq chat and transcription configuration does not automatically provide TTS. Configure a separate supported speech-generation service.
- **Tavily search unavailable:** Check `TAVILY_API_KEY`. External search is optional and will not work without a valid key.
- **Camera and microphone:** Grant browser permissions and use localhost or a properly configured HTTPS deployment.

## Security and Medical Limitations

This project is an educational software scaffold, not a production medical device.

Before deploying it for real users:

- Replace a shared API token with proper authentication and authorization.
- Add request limits, secure file validation, logging, and error monitoring.
- Implement encrypted conversation storage and clear data-retention and deletion controls.
- Review privacy, security, legal, and licensing requirements before processing patient data or hosting/indexing medical reference books.
- Use current, authoritative clinical references and clinically reviewed emergency guidance.
- Evaluate retrieval quality, model responses, and failure modes with appropriate testing.
- Secure production deployments behind HTTPS and avoid exposing API keys in browser code.

Medical references may contain outdated advice. Model instructions cannot guarantee clinical safety. Image descriptions are observations, not diagnoses, and cannot confirm or rule out a disease. Users should consult qualified healthcare professionals for medical decisions and seek emergency assistance for emergencies.