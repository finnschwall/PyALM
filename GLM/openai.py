
import openai  # for OpenAI API calls
import time 
import os

class OpenAI(GLM):

    def __init__(self,model_path_or_name, openai_key=None):
        super().__init__(model_path_or_name, n_ctx=n_ctx, verbose=verbose)
        if openai_key:
            openai.api_key = openai_key
            if not "OPENAI_API_KEY" in os.environ:
                raise Exception("No openai key set!")
        conv = {"chatgpt":"gpt-3.5-turbo", "gpt4":"gpt-4"}