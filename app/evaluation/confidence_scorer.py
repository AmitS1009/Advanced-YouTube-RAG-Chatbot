from typing import List
from langchain_core.documents import Document
from app.config.settings import settings
from app.utils.logger import setup_logger

from langsmith import traceable

logger = setup_logger(__name__)

class ConfidenceScorer:
    @staticmethod
    @traceable(name="confidence_scoring", run_type="tool")
    def calculate_confidence(documents: List[Document]) -> str:
        """
        Evaluates confidence based on retrieval scores.
        Returns: "HIGH", "MEDIUM", "LOW"
        """
        if not documents:
            return "LOW"
            
        # Check top document score
        # CrossEncoder scores usually involve logits, but often > 0 is good, < 0 is bad.
        # Or if normalized 0-1.
        # Assuming ms-marco-MiniLM-L-6-v2 which yields unbounded logits roughly -10 to 10.
        # We might need to sigmoid it or just use a raw threshold.
        # For this specific model: > 0 is generally relevant.
        # Let's say: 
        # > 3: High
        # 0 - 3: Medium
        # < 0: Low (Prune?)
        
        # However, user said "If similarity score < threshold (e.g. 0.3)"
        # If we rely on cosine similarity (0-1), 0.3 is very low.
        # We stored 'relevance_score' in metadata.
        
        top_score = documents[0].metadata.get('relevance_score', 0.0)
        logger.info(f"Top relevance score: {top_score}")
        
        # Thresholds (Tunable)
        HIGH_THRESHOLD = 2.0
        MEDIUM_THRESHOLD = 0.0
        
        if top_score > 0.5:
            return "HIGH"
        elif top_score > -2.0:
            return "MEDIUM"
        else:
            return "LOW"
