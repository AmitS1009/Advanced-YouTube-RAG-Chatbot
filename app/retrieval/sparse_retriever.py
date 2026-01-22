import pickle
import os
from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from app.config.settings import settings
from app.utils.logger import setup_logger
from langsmith import traceable

logger = setup_logger(__name__)

class SparseRetriever:
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.index_path = os.path.join(settings.DATA_DIR, "bm25_index.pkl")
        self.load_index()

    def create_index(self, documents: List[Document]):
        """
        Creates BM25 index from documents.
        documents: List of LangChain Documents.
        """
        logger.info(f"Creating BM25 index with {len(documents)} documents...")
        self.documents = documents
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save_index()

    def save_index(self):
        with open(self.index_path, "wb") as f:
            pickle.dump((self.bm25, self.documents), f)
        logger.info("BM25 index saved.")

    def load_index(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    self.bm25, self.documents = pickle.load(f)
                logger.info("BM25 index loaded.")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
                self.bm25 = None
        else:
            logger.info("No BM25 index found.")

    @traceable(name="sparse_retrieval", run_type="retriever")
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        if not self.bm25 or not self.documents:
            logger.warning("BM25 index is empty.")
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = self.bm25.get_top_n(tokenized_query, self.documents, n=top_k)
        
        # We need to map back to get scores for these top_n
        # This is a bit inefficient with rank_bm25 unmodified, so we do it manually
        doc_scores = []
        # Zip all scores with docs
        all_doc_scores = zip(self.documents, scores)
        # Sort by score desc
        sorted_docs = sorted(all_doc_scores, key=lambda x: x[1], reverse=True)
        
        return sorted_docs[:top_k]
