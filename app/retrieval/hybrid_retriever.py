from typing import List
from langchain_core.documents import Document
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.reranker import Reranker
from app.utils.logger import setup_logger
from app.config.settings import settings

logger = setup_logger(__name__)

class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever, reranker: Reranker):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.reranker = reranker

    def search(self, query: str) -> List[Document]:
        """
        Executes hybrid search: Dense + Sparse -> Merge -> Rerank.
        """
        # 1. Retrieve
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
