from astrbot.api.star import Context
from astrbot.api import logger
from astrbot.core.provider.entities import LLMResponse

class LLMService:
    """
    一个服务类，用于封装对 AstrBot LLM Provider 的调用。
    """
    def __init__(self, context: Context, provider_id: str = None):
        self.context = context
        self.provider_id = provider_id

    async def generate(self, user_prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        使用指定的或默认的 LLM Provider 生成文本。
        """
        provider = None
        if self.provider_id:
            provider = self.context.get_provider_by_id(self.provider_id)
            if not provider:
                logger.warning(f"LLM Provider with ID '{self.provider_id}' not found. Falling back to default.")

        if not provider:
            provider = self.context.get_using_provider()

        if not provider:
            logger.error("No available LLM Provider found.")
            return "Error: No LLM provider available."

        try:
            # The correct method is text_chat.
            response: LLMResponse = await provider.text_chat(
                prompt=user_prompt,
                system_prompt=system_prompt,
                **kwargs
            )
            # Per the entity definition, using result_chain is the recommended approach.
            if response.result_chain:
                return response.result_chain.get_plain_text()
            # Fallback for safety, though result_chain should be present.
            return response.completion_text
        except Exception as e:
            provider_name = self.provider_id or "default"
            logger.error(f"Error calling LLM Provider '{provider_name}': {e}", exc_info=True)
            return f"Error during LLM call: {e}"

class EmbeddingService:
    """
    一个服务类，用于封装对 AstrBot Embedding Provider 的调用。
    """
    def __init__(self, context: Context, provider_id: str = None):
        self.context = context
        self.provider_id = provider_id

    async def get_embedding(self, text: str, **kwargs) -> list[float]:
        """
        使用指定的或默认的 Embedding Provider 生成嵌入向量。
        """
        provider = None
        if self.provider_id:
            provider = self.context.get_provider_by_id(self.provider_id)
            if not provider:
                logger.warning(f"Embedding Provider with ID '{self.provider_id}' not found. Falling back to default.")
        
        if not provider:
            providers = self.context.get_all_embedding_providers()
            if providers:
                provider = providers[0]

        if not provider:
            logger.error("No available Embedding Provider found.")
            return []

        try:
            # get_embeddings 是一个批量方法，我们传递一个单元素的列表并取回第一个结果
            embeddings = await provider.get_embeddings([text], **kwargs)
            return embeddings[0] if embeddings else []
        except Exception as e:
            provider_name = self.provider_id or "default"
            logger.error(f"Error calling Embedding Provider '{provider_name}': {e}", exc_info=True)
            return []