class QuerytoChatTemplateTransformer:
    def to_chat_format(self, query: str) -> str:
        """
        사용자의 query를 LLM의 채팅 형식으로 변환하는 로직.
        """
        # 간단한 예시로, 사용자 입력을 챗 형식으로 변환
        return f"[USER]: {query}\n[LLM]:"

class QueryPreprocessor:
    # TODO : 필요시
    def preprocess(self, query: str) -> str:
        return query


def get_llm_config(settings)->dict:
    if settings.APP_ENVIRONMENT == "prototype":
        llm_host = settings.OLLAMA_API_HOST
        llm_name = settings.OLLAMA_MODEL_NAME

    elif settings.APP_ENVIRONMENT == "development":
        llm_host = settings.VLLM_API_HOST
        llm_name = settings.VLLM_MODEL_NAME

    else:
        raise ValueError(f"Unknown APP_ENVIRONMENT: {settings.APP_ENVIRONMENT}")
    
    return {
        "llm_host": llm_host,
        "llm_name": llm_name
    }
