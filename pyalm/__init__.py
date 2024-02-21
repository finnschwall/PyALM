def __getattr__(name):
    if name.upper() == "LLAMA":
        from .llama import LLaMa
        return LLaMa
    if name == "OpenAI":
        from .openai import OpenAI
        return OpenAI
    if name.lower() == "alephalpha":
        from .alephalpha import AlephAlpha
        return AlephAlpha
    if name.lower() == "gemini":
        from .gemini import Gemini
        return Gemini
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from .alm import ConversationRoles, ALM