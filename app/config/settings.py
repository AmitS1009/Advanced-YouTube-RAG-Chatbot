import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
    CHUNKS_DIR = os.path.join(DATA_DIR, "processed_chunks")
    VECTORSTORE_DIR = os.path.join(DATA_DIR, "faiss_index")

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Models
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
    GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    # RAG Parameters
    CHUNK_SIZE = 500  # characters
    CHUNK_OVERLAP = 100 # characters, approx 20-30%
    TIME_WINDOW_SECONDS = 180 # 3 minutes
    
    RETRIEVAL_TOP_K = 10
    RERANK_TOP_K = 4
    
    SIMILARITY_THRESHOLD = 0.3

settings = Settings()
