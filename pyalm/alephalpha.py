from .alm import ALM
import time
import os
from aleph_alpha_client import Client, CompletionRequest, Prompt
from functools import partial


class OpenAI(ALM):

    def __init__(self, model_path_or_name, aleph_alpha_key=None, verbose=0, n_ctx=2048, **kwargs):
        super().__init__(model_path_or_name, n_ctx=n_ctx, verbose=verbose)
        if aleph_alpha_key:
            self.api_key = aleph_alpha_key
        elif "AA_TOKEN" in os.environ:
            self.api_key =os.getenv("AA_TOKEN")
        elif not "AA_TOKEN" in os.environ:
            raise Exception("No aleph_alpha_key key set!")

        self.client = Client(token=self.api_key)

        self.finish_meta = {}
        self.pricing = {"gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}}

    # @abstractmethod
    def tokenize(self, text):
        encoding = tiktoken.encoding_for_model(self.model_path_or_name)
        return encoding.encode(text)

    def tokenize_as_str(self, text):
        encoding = tiktoken.encoding_for_model(self.model_path_or_name)
        encoded = encoding.encode(text)
        return [encoding.decode_single_token_bytes(token).decode("utf-8") for token in encoded]

    def get_n_tokens(self, text):
        return len(self.tokenize(text))

    def _extract_message_from_generator(self, gen):

        for i in gen:
            try:
                token = i["choices"][0]["delta"]["content"]
            except:
                self.finish_meta["finish_reason"] = i["choices"][0]["finish_reason"]
            print(token, end ="")
            # self.test_txt += token
            yield token, None

    def create_native_generator(self, text, stream=True, keep_dict=False,token_prob_delta = None,
                                token_prob_abs = None, **kwargs):
        if token_prob_abs:
            raise Exception("OpenAI API only supports relative logit chance change")
        if token_prob_delta:
            response = openai.ChatCompletion.create(
                model=self.model_path_or_name,
                messages=text,
                stream=stream,
                logit_bias = token_prob_delta,
                **kwargs
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.model_path_or_name,
                messages=text,
                stream=stream,
                **kwargs
            )

        if keep_dict:
            return response
        else:
            if stream:
                return self._extract_message_from_generator(response, stream=stream)
            else:
                return response["choices"][0]["message"]["content"]

    def build_prompt(self):
        prompt = []
        if "content" in self.system_msg:
            prompt.append({"role": self.symbols["SYSTEM"], "content":
                            self._replace_symbols(self.system_msg["content"])})

        for i in self.conv_history:
            prompt.append({"role": self.symbols[str(i["role"])], "content":  self._replace_symbols(i["content"])})
        return prompt
