from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm.llm_client import LLMClient
from app.config.prompts import QUERY_REWRITE_PROMPT
from app.utils.logger import setup_logger

from langsmith import traceable

logger = setup_logger(__name__)

class QueryRewriter:
    def __init__(self):
        self.llm = LLMClient().get_model()
        # self.prompt = PromptTemplate.from_template(QUERY_REWRITE_PROMPT) # This is no longer needed if prompt is constructed dynamically

    @traceable(name="query_rewrite", run_type="chain")
    def rewrite(self, query: str, chat_history: list = None) -> str:
        """
        Rewrites the user query to be search-optimized, using history for context.
        """
        history_str = ""
        if chat_history:
            # Format last 3 turns
            recent = chat_history[-3:]
            history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent])

        prompt_template_str = (
            f"You are an expert search query optimizer.\n"
            f"Chat History:\n{history_str}\n\n"
            f"Original Query: {query}\n"
            f"Task: Rewrite the query to be an effective keyword search for a transcript.\n"
            f"Rules:\n"
            f"1. Resolve pronouns ('it', 'he', 'that') using Chat History.\n"
            f"2. If the query asks about a specific named entity (e.g. 'AlphaFold', 'Demis'), KEEP IT EXPLICIT in the rewrite.\n"
            f"3. Remove conversational filler ('what is', 'tell me about').\n"
            f"4. Add 1-2 relevant synonyms or context keywords ONLY if the query is ambiguous.\n"
            f"5. If the query is already a specific keyword, return it AS IS.\n"
            f"Output ONLY the rewritten query text."
        )
        logger.info(f"Rewriting query: {query}")
        try:
            # Create a PromptTemplate on the fly for the dynamic prompt string
            dynamic_prompt = PromptTemplate.from_template(prompt_template_str)
            chain = dynamic_prompt | self.llm | StrOutputParser()
            # Since the query is already embedded in the prompt_template_str, invoke with an empty dictionary or None
            rewritten_query = chain.invoke({})
            logger.info(f"Rewritten query: {rewritten_query}")
            return rewritten_query.strip()
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return query
