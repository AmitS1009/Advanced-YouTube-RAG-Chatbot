from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from app.config.settings import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class LLMClient:
    def __init__(self, provider: str = None):
        self.provider = provider or settings.LLM_PROVIDER
        self.model = self._initialize_model()

    def _initialize_model(self) -> BaseChatModel:
        if self.provider == "groq":
            if not settings.GROQ_API_KEY:
                logger.error("GROQ_API_KEY not found.")
                raise ValueError("GROQ_API_KEY not found.")
            logger.info(f"Initializing Groq model: {settings.GROQ_MODEL}")
            return ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.GROQ_MODEL,
                temperature=0.0
            )
        elif self.provider == "gemini":
            if not settings.GOOGLE_API_KEY:
                logger.error("GOOGLE_API_KEY not found.")
                raise ValueError("GOOGLE_API_KEY not found.")
            logger.info(f"Initializing Gemini model: {settings.GEMINI_MODEL}")
            return ChatGoogleGenerativeAI(
                google_api_key=settings.GOOGLE_API_KEY,
                model=settings.GEMINI_MODEL,
                temperature=0.0
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_model(self) -> BaseChatModel:
        return self.model
