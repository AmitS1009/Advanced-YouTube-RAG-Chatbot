from typing import List
from langchain_core.documents import Document
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.reranker import Reranker
from app.utils.logger import setup_logger
from app.config.settings import settings

from langsmith import traceable

logger = setup_logger(__name__)

class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever, reranker: Reranker):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.reranker = reranker

    @traceable(name="hybrid_search_pipeline", run_type="chain")
    def search(self, query: str) -> List[Document]:
        """
        Executes hybrid search: Dense + Sparse -> Merge -> Rerank.
        """
        # 1. Check for Summarization Intent (Heuristic)
        if "summar" in query.lower() or "overview" in query.lower():
            logger.info("Summarization intent detected. Fetching broad context distribution.")
            # Strategy: Get chunks from beginning, middle, and end.
            # Dense retriever usually finds *similar* chunks, which might be repetitive intro/outro.
            # Sparse is better for keywords.
            # But the best way is essentially ignoring search and grabbing spread-out chunks.
            # For this simple implementation, we'll force retrieving a large number of top chunks
            # and hope they cover different parts, OR we rely on the fact that "summary" might match abstract sections.
            
            # Better approach: Just use standard retrieval but with much higher K to cover ground.
            dense_results = self.dense.retrieve(query, top_k=50)
            sparse_results = self.sparse.retrieve(query, top_k=50)
        else:
            # Standard Retrieval
            dense_results = self.dense.retrieve(query, top_k=settings.RETRIEVAL_TOP_K)
            sparse_results = self.sparse.retrieve(query, top_k=settings.RETRIEVAL_TOP_K)
        
        # 2. Merge (Deduplicate based on content or ID)
        # Note: Since documents are objects, we use content as a hash for simple dedup
        seen_content = set()
        merged_docs = []
        
        # Simple interleave or score merge could be better, but appending unique is fine for reranker
        for doc, score in dense_results:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                merged_docs.append(doc)
        
        for doc, score in sparse_results:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                merged_docs.append(doc)
                
        logger.info(f"Merged {len(merged_docs)} unique documents.")

        # 3. Rerank
        final_docs = self.reranker.rerank(query, merged_docs, top_k=settings.RERANK_TOP_K)
        
        return final_docs
