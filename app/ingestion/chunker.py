from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.settings import settings
from app.utils.logger import setup_logger

from langsmith import traceable

logger = setup_logger(__name__)

class TimeAwareChunker:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.window_size = settings.TIME_WINDOW_SECONDS

    @traceable(name="create_chunks", run_type="tool")
    def create_chunks(self, transcript_items: List[Dict], video_id: str) -> List[Dict[str, Any]]:
        """
        Takes raw transcript items [{'text':..., 'start':..., 'duration':...}]
        and returns enriched chunks.
        
        Strategy:
        1. Group transcript items into time windows (e.g. 3 minutes).
        2. Within each window, join text and split using RecursiveCharacterTextSplitter.
        3. Assign metadata (start_time, end_time, video_id) to each chunk.
        """
        if not transcript_items:
            return []

        chunks = []
        
        # 1. Group by time windows
        current_window_start = 0.0
        current_window_text = []
        current_window_items = []
        
        for item in transcript_items:
            start = item['start']
            text = item['text']
            
            if start >= current_window_start + self.window_size:
                # Process the completed window
                window_chunks = self._process_window(
                    current_window_text, 
                    current_window_start, 
                    video_id
                )
                chunks.extend(window_chunks)
                
                # Start new window
                current_window_start = start
                current_window_text = [text]
                current_window_items = [item]
            else:
                current_window_text.append(text)
                current_window_items.append(item)
        
        # Process the final pending window
        if current_window_text:
             window_chunks = self._process_window(
                current_window_text, 
                current_window_start, 
                video_id
            )
             chunks.extend(window_chunks)

        logger.info(f"Created {len(chunks)} chunks for video {video_id}")
        return chunks

    def _process_window(self, text_list: List[str], window_start_time: float, video_id: str) -> List[Dict]:
        """
        Internal method to split text within a timeframe and assign metadata.
        """
        full_text = " ".join(text_list)
        
        # Split text (Semantic split)
        # Note: This is an approximation. 
        # Ideally we would map the split text back to exact timestamps of the words,
        # but for a RAG chatbot, a "Window Start Time" is often sufficient 
        # to jump to the general area.
        
        split_docs = self.text_splitter.create_documents([full_text])
        
        enriched_chunks = []
        for i, doc in enumerate(split_docs):
            chunk_metadata = {
                "video_id": video_id,
                "window_start_time": window_start_time, # approximate start
                "chunk_index": i,
                "text": doc.page_content
            }
            enriched_chunks.append(chunk_metadata)
            
        return enriched_chunks
