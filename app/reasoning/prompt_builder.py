from typing import List
from langchain_core.documents import Document
from app.config.prompts import ANSWER_GENERATOR_SYSTEM_PROMPT

class PromptBuilder:
    @staticmethod
    def build_system_message() -> str:
        return ANSWER_GENERATOR_SYSTEM_PROMPT

    @staticmethod
    def build_context_string(documents: List[Document]) -> str:
        """
        Formats documents into a context string with metadata.
        """
        context_parts = []
        for i, doc in enumerate(documents):
            # Format: [ID] (Time: MM:SS) Content...
            offset = doc.metadata.get('window_start_time', 0)
            minutes = int(offset // 60)
            seconds = int(offset % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            
            content = f"Chunk {i+1} (Start: {timestamp}):\n{doc.page_content}\n"
            context_parts.append(content)
            
        return "\n".join(context_parts)
