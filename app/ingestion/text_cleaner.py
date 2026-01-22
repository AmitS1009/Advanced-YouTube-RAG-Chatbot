import re
from app.utils.logger import setup_logger

from langsmith import traceable

logger = setup_logger(__name__)

class TextCleaner:
    @traceable(name="clean_text", run_type="tool")
    def clean_text(self, text: str) -> str:
        """
        Aggressive cleaning of the transcript text.
        1. Remove fillers (um, uh, you know) - careful not to remove semantic ones.
        2. Remove bracketed noise like [Music], [Applause].
        3. Normalize whitespace.
        """
        # Remove bracketed content, e.g., [Music], (Laughter)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        
        # Remove common fillers (basic regex, can be improved)
        # Note: Be careful with "you know" as it can be part of a valid sentence.
        # We will mostly target standalone vocal fillers or obvious noise.
        fillers = [
            r'\buh\b', r'\bum\b', r'\buh-huh\b', 
            # r'\byou know\b', # risky
        ]
        
        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)

        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
