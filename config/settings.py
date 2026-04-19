import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_BASE_URL: str = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    LLM_MODEL: str = os.getenv("CHAT_MODEL", "qwen3.5-plus")
    ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "qwen-plus")  # lightweight, no thinking
    RETRIEVAL_MODEL: str = os.getenv("RETRIEVAL_MODEL", "qwen-plus")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    FOOTBALL_API_KEY: str = os.getenv("FOOTBALL_API_KEY", "")
    TEMPERATURE: float = 0.3
    NO_THINKING: dict = {"extra_body": {"enable_thinking": False}}
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    KNOWLEDGE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge")

    # LangSmith Tracing (optional — set LANGSMITH_API_KEY to enable)
    LANGSMITH_TRACING: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "FootballGPT")


settings = Settings()
