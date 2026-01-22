import os
import shutil
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.embeddings.embedding_model import EmbeddingModel
from app.config.settings import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class FaissVectorStore:
    def __init__(self):
        self.embeddings = EmbeddingModel.get_embedding_model()
        self.index_path = settings.VECTORSTORE_DIR
        self.vector_store: Optional[FAISS] = None
        self.load_index()

    def load_index(self):
        """Loads the FAISS index from disk if it exists."""
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            logger.info("Loading existing FAISS index...")
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True # Trusted local source
                )
                logger.info("FAISS index loaded.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.vector_store = None
        else:
            logger.info("No existing FAISS index found.")
            self.vector_store = None

    def create_index(self, chunks: List[dict]):
        """
        Creates a new FAISS index from enriched chunks.
        chunks: List of specific metadata dicts from chunker.
        """
        if not chunks:
            logger.warning("No chunks provided to create index.")
            return

        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk['text'],
                metadata={k: v for k, v in chunk.items() if k != 'text'}
            )
            documents.append(doc)

        logger.info(f"Creating FAISS index with {len(documents)} documents...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.save_index()

    def add_documents(self, chunks: List[dict]):
        """Adds documents to existing index or creates new one."""
        if self.vector_store is None:
            self.create_index(chunks)
        else:
            documents = [
                Document(page_content=c['text'], metadata={k: v for k, v in c.items() if k != 'text'}) 
                for c in chunks
            ]
            self.vector_store.add_documents(documents)
            self.save_index()

    def save_index(self):
        """Saves the index to disk."""
        if self.vector_store:
            if not os.path.exists(self.index_path):
                os.makedirs(self.index_path)
            self.vector_store.save_local(self.index_path)
            logger.info(f"FAISS index saved to {self.index_path}")

    def as_retriever(self, search_kwargs: dict = None):
        if not self.vector_store:
            logger.exception("Vector store is empty!")
            return None
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs or {})
