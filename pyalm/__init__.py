# TODO fix this awful naming convention
# def __getattr__(name):
#     if name.upper() == "LLAMA":
#         from .llama import LLaMa
#         return LLaMa
#     if name.lower() == "alephalpha":
#         from .alephalpha import AlephAlpha
#         return AlephAlpha
#     if name.lower() == "gemini":
#         from .gemini import Gemini
#         return Gemini
#     raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# from .openai import OpenAI
from .alm import ConversationRoles, ALM
from .state import ConversationRoles, ConversationTracker
from .resources import get_gpu_usage