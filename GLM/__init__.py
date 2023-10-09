def __getattr__(name):
    if name.upper() == "LLAMA":
        from .llama import LLaMa
        return LLaMa
    if name.upper() == "OPENAI":
        from .openai import OpenAI
        return OpenAI
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from .glm import ConversationRoles