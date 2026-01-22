from typing import List, Tuple
from langchain_core.documents import Document
from app.vectorstore.faiss_store import FaissVectorStore
from app.utils.logger import setup_logger

from langsmith import traceable

logger = setup_logger(__name__)

class DenseRetriever:
    def __init__(self, vector_store: FaissVectorStore):
        self.vector_store = vector_store

    @traceable(name="dense_retrieval", run_type="retriever")
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Retrieves documents using vector similarity.
        Returns list of (Document, score).
        Note: FAISS scores are L2 distances (lower is better) if using default, 
        or Inner Product (higher is better) if normalized.
        LangChain's similarity_search_with_score usually returns L2.
        """
        if not self.vector_store.vector_store:
            logger.warning("Vector store not initialized.")
            return []
            
        logger.info(f"Dense retrieval for: {query}")
        results = self.vector_store.vector_store.similarity_search_with_score(query, k=top_k)
        
        # Normalize scores if needed? 
        # For L2, lower is better. We might want to convert to similarity 0-1.
        # But for now, we pass raw scores.
        return results
