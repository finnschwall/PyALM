from .glm import GLM
import openai
import time 
import os
import tiktoken
from functools import partial

class OpenAI(GLM):

    def __init__(self,model_path_or_name, openai_key=None, verbose = 0, n_ctx=2048, **kwargs):
        super().__init__(model_path_or_name, n_ctx=n_ctx, verbose=verbose)
        if openai_key:
            openai.api_key = openai_key
        elif not "OPENAI_API_KEY" in os.environ:
            raise Exception("No openai key set!")
                
        conv = {"gpt3": "gpt-3.5-turbo","gpt-3": "gpt-3.5-turbo", "chatgpt":"gpt-3.5-turbo", "gpt4":"gpt-4", "gpt-3.5-turbo"}
        self.model_path_or_name = conv.get(model_path_or_name, model_path_or_name)
        self.symbols["ASSISTANT"] = "assistant"
        self.symbols["USER"] = "user"
        self.finish_meta  = {}
        self.pricing = {"gpt-3.5-turbo":{"input": 0.0015, "output": 0.002},
                        "gpt-3.5-turbo-16k":{"input": 0.003 , "output": 0.004},
                        "gpt-4":{"input": 0.03 , "output": 0.06}}

    # @abstractmethod
    def tokenize(self, text):
        encoding = tiktoken.encoding_for_model(self.model_path_or_name)
        return encoding.encode(text)

    def tokenize_as_str(self, text):
        encoding = tiktoken.encoding_for_model(self.model_path_or_name)
        encoded =  encoding.encode(text)
        return [encoding.decode_single_token_bytes(token) for token in encoded]

        
    def get_n_tokens(self, text):
        return len(self.tokenize(text))
    
    def _extract_message_from_generator(self, gen):
        for i in gen:
            try:
                token= i["choices"][0]["delta"]["content"]
            except:
                self.finish_meta["finish_reason"] = i["choices"][0]["finish_reason"]
            yield token
        

    def create_native_generator(self, text, keep_dict=False, **kwargs):
        response = openai.ChatCompletion.create(
            model=self.model_path_or_name,
            messages= text,
            stream=True,
            **kwargs
        )
        if keep_dict:
            return response
        else:
            return self._extract_message_from_generator(response)


    def build_prompt(self):
        prompt = []
        for i in self.conv_history:
            prompt.append({"role": self.symbols[str(i["role"])], "content":i["content"]})
        return prompt
