from typing import Generator
import time
from langchain_core.messages import SystemMessage, HumanMessage
from app.llm.llm_client import LLMClient
from app.reasoning.prompt_builder import PromptBuilder
from app.evaluation.answer_validator import AnswerValidator
from app.utils.logger import setup_logger

from langsmith import traceable

logger = setup_logger(__name__)

class AnswerGenerator:
    def __init__(self):
        self.llm = LLMClient().get_model()
        self.validator = AnswerValidator()

    @traceable(name="generate_answer", run_type="chain")
    def generate_answer(self, query: str, documents: list) -> Generator[str, None, None]:
        """
        Generates answer from documents with validation.
        Yields chunks for streaming.
        
        Note: Validation requires the FULL answer. So we must generate first, validate, 
        and then yield (or yield as we go and revoke? Streaming + Validation is tricky).
        
        Strategy:
        1. Generate full answer first (internal).
        2. Validate.
        3. If valid, yield chunks (simulated streaming or just yield the full text if real streaming is impossible after generation).
        
        Wait, user wants "make answer streaming".
        Real-time validation is hard. 
        Option A: Stream chunks, then at the end validating. If invalid, append a disclaimer.
        Option B: Generate full, validate, then stream the specific string.
        
        Given the strict "Reject answer" instruction, Option B is safer.
        """
        if not documents:
            yield "No relevant context found in this video."
            return

        context_str = PromptBuilder.build_context_string(documents)
        system_msg = PromptBuilder.build_system_message()
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {query}")
        ]
        
        logger.info("Generating answer...")
        full_response = ""
        
        try:
            # We generate fully first to validate
            response_msg = self.llm.invoke(messages)
            full_response = response_msg.content
            
            # Post-Processing: Formatting
            # User wants new line from one timestamp to another
            # Pattern: [MM:SS] or (Start: MM:SS)
            # We want to ensure there is a \n\n before/after citations if they mark section ends.
            # Brute force: Replace "] " with "]\n\n" if it's likely a sentence end.
            # Or better: Just replace all [MM:SS] with [MM:SS]\n\n if followed by text.
            
            # Simple fix: Replace any occurrence of timestamp citation with itself + double newline
            # But only if it's the end of a thought? 
            # User said "start from new line after content from one time stamp gets end".
            # Usually the model puts citation at the end of the sentence.
            # So [MM:SS].\n\n
            
            import re
            # Add double newline after citations.
            # Matches: [12:34], (Start: 12:34), (Chunk 1, Start: 12:34)
            # Regex explanation:
            # (\[\d{2}:\d{2}\]|\(.*?Start: \d{2,}:\d{2}.*?\))
            # Catch standard [MM:SS] OR (...) containing "Start: MM:SS"
            
            full_response = re.sub(r'(\[\d{2}:\d{2}\]|\(.*?\d{2,}:\d{2}.*?\))', r'\1\n\n', full_response)
            # Remove triple newlines if any created
            full_response = full_response.replace("\n\n\n", "\n\n")
            
            # Validation
            is_valid = self.validator.validate(query, full_response, context_str)
            
            if not is_valid:
                logger.warning("Answer validation failed.")
                yield "This information is not clearly present in the video (derived from strict validation)."
            else:
                # Mock streaming the trusted response
                # (Since we have the full text, we can yield words)
                # True Streaming Simulation
                # Yield 1 word at a time with a tiny delay
                words = full_response.split(" ")
                for word in words:
                    yield word + " "
                    # Dynamic delay: punctuations take longer to "type"
                    if word.endswith(('.', '?', '!')):
                         time.sleep(0.015)
                    else:
                         time.sleep(0.005)
                    
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            yield "An error occurred while generating the answer."
