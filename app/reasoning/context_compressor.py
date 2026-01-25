from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm.llm_client import LLMClient
from app.config.prompts import CONTEXT_COMPRESSOR_PROMPT
from app.utils.logger import setup_logger

from langsmith import traceable

logger = setup_logger(__name__)

class ContextCompressor:
    def __init__(self):
        self.llm = LLMClient().get_model()
        self.prompt = PromptTemplate.from_template(CONTEXT_COMPRESSOR_PROMPT)

    @traceable(name="context_compression", run_type="chain")
    def compress(self, documents: List[Document]) -> List[Document]:
        """
        Summarizes each document to reduce noise.
        """
        compressed_docs = []
        logger.info(f"Compressing {len(documents)} documents...")
        
        # bypass compression for now to preserve details
        logger.info("Compression disabled to preserve details.")
        return documents
