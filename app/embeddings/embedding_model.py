from langchain_huggingface import HuggingFaceEmbeddings
from app.config.settings import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingModel:
    _instance = None

    @classmethod
    def get_embedding_model(cls):
        """
        Singleton pattern to load the embedding model once.
        """
        if cls._instance is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            cls._instance = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}, # Force CPU for compatibility, change to 'cuda' if available
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model loaded successfully.")
        return cls._instance
