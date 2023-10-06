def __getattr__(name):
    if name == "LLaMa":
        from .llama import LLaMa
        return LLaMa
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")