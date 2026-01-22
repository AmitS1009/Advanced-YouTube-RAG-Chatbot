from typing import List
from app.embeddings.embedding_model import EmbeddingModel
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class Embedder:
    def __init__(self):
        self.model = EmbeddingModel.get_embedding_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.
        """
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query.
        """
        return self.model.embed_query(text)
