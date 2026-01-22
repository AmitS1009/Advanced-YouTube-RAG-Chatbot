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
        
        chain = self.prompt | self.llm | StrOutputParser()
        
        # We can run this in parallel for speed, but sequential for now to avoid rate limits
        for doc in documents:
            try:
                summary = chain.invoke({"context": doc.page_content})
                # Create a new doc with summary as content, keeping metadata
                new_doc = Document(page_content=summary, metadata=doc.metadata)
                # Store original content in metadata for citation if needed, 
                # but instruction says "Keep original chunk only for citation" -> usually means metadata
                new_doc.metadata['original_text'] = doc.page_content
                compressed_docs.append(new_doc)
            except Exception as e:
                logger.error(f"Error compressing doc: {e}")
                compressed_docs.append(doc) # Fallback to original
                
        return compressed_docs
