from pyalm.internal.alm import ALM
import os
from timeit import default_timer as timer

import llama_cpp


class LLaMa(ALM):

    def __init__(self, model_path_or_name, verbose=0, **kwargs):
        super().__init__(model_path_or_name, verbose=verbose, **kwargs)
        self.model_name = model_path_or_name.split("/")[-1]
        self.model = llama_cpp.Llama(model_path_or_name, verbose=verbose, **kwargs)

        llama_specifics = {"ASSISTANT": "assistant", "USER": "user", "SYSTEM": "system"}
        self._built_in_symbols.update(llama_specifics)



    def create_native_completion(self, text, temp=0, **kwargs):
        if isinstance(text, str):
            return self.model.create_completion(text, temperature=temp,**kwargs)
        else:
            answer = self.model.create_chat_completion(text, temperature=temp, **kwargs)
            self.finish_meta["tokens"] = answer["usage"]
            return answer["choices"][0]["message"]["content"]


    def _extract_message_from_generator(self, gen):
        for i in gen:
            token = i["choices"][0]["text"]
            finish_reason = i["choices"][0]["finish_reason"]
            if finish_reason:
                self.finish_meta["finish_reason"] = finish_reason
            if token is None:
                continue
            yield token, None

    def create_native_generator(self, text, **kwargs):
        if isinstance(text, str):
            gen = self.model.create_completion(text, stream=True, **kwargs)
        else:
            gen = self.model.create_chat_completion(text, stream=True, **kwargs)
        return self._extract_message_from_generator(gen)

    def get_n_tokens(self, text):
        return len(self.model.tokenize(bytes(text, "utf-8")))
