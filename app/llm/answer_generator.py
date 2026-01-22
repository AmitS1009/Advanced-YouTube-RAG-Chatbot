from typing import Generator
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
            
            # Validation
            is_valid = self.validator.validate(query, full_response, context_str)
            
            if not is_valid:
                logger.warning("Answer validation failed.")
                yield "This information is not clearly present in the video (derived from strict validation)."
            else:
                # Mock streaming the trusted response
                # (Since we have the full text, we can yield words)
                chunk_size = 5 
                words = full_response.split()
                for i in range(0, len(words), chunk_size):
                    yield " ".join(words[i:i+chunk_size]) + " "
                    
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            yield "An error occurred while generating the answer."
