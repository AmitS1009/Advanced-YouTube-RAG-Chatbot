from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from app.config.settings import settings
from app.utils.logger import setup_logger

from langsmith import traceable

logger = setup_logger(__name__)

class Reranker:
    def __init__(self):
        # Lightweight cross encoder
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    @traceable(name="rerank", run_type="retriever")
    def rerank(self, query: str, documents: List[Document], top_k: int = 4) -> List[Document]:
        """
        Reranks a list of documents based on relevance to the query.
        """
        if not documents:
            return []
            
        logger.info(f"Reranking {len(documents)} documents...")
        
        # Prepare pairs [query, doc_text]
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Combine docs with scores
        doc_scores = list(zip(documents, scores))
        
        # Sort by score descending
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k docs with scores in metadata
        final_docs = []
        for doc, score in doc_scores[:top_k]:
            doc.metadata['relevance_score'] = float(score)
            final_docs.append(doc)
            
        return final_docs
