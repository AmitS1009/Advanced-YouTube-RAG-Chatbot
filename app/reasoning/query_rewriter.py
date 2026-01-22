from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm.llm_client import LLMClient
from app.config.prompts import QUERY_REWRITE_PROMPT
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class QueryRewriter:
    def __init__(self):
        self.llm = LLMClient().get_model()
        # self.prompt = PromptTemplate.from_template(QUERY_REWRITE_PROMPT) # This is no longer needed if prompt is constructed dynamically

    def rewrite(self, query: str) -> str:
        """
        Rewrites the user query to be search-optimized.
        """
        prompt_template_str = (
            f"You are a helpful assistant that rewrites user queries for better retrieval from a YouTube transcript.\n"
            f"Original Query: {query}\n"
            f"Rewrite this query to be more specific, keyword-rich, and suitable for semantic search.\n"
            f"IMPORTANT: Output ONLY the rewritten query text. Do not include any explanations, prefixes, or quotes."
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
