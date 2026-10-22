import os
import certifi 
from dotenv import load_dotenv
load_dotenv()

# --- AUTOMATIC SSL FIX FOR CONDA/PYTHON ENVIRONMENTS ---
ssl_cert = os.environ.get('SSL_CERT_FILE')
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ['SSL_CERT_FILE'] = certifi.where()
elif not ssl_cert:
    os.environ['SSL_CERT_FILE'] = certifi.where()
# --------------------------------------------------------

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GROQ_API_KEY:str = ''
    TAVILY_API_KEY:str = ''
    PINECONE_API_KEY:str = ''
    PINECONE_INDEX:str = 'agenticmedibot'
    GROQ_CHAT_MODEL:str = 'openai/gpt-oss-20b'
    EMBED_MODEL:str = 'sentence-transformers/all-MiniLM-L6-v2'
    GROQ_VISION_MODEL:str = 'qwen/qwen3.8-27b'
    API_TOKEN:str = ''
    ALLOWED_ORIGIN:str = 'http://localhost:8000,https://127.0.0.1:8000'
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()
