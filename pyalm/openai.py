from .alm import ALM
import openai
import time
import os
import tiktoken
from functools import partial
from warnings import warn
from timeit import default_timer as timer


class OpenAI(ALM):
    available_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]
    pricing = {"gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
               "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
               "gpt-4": {"input": 0.03, "output": 0.06}, }
    pricing_meta = {"currency": "$", "token_unit": 1000}

    def __init__(self, model_path_or_name, openai_key=None, verbose=0, n_ctx=2048, **kwargs):
        super().__init__(model_path_or_name, n_ctx=n_ctx, verbose=verbose)
        if openai_key:
            openai.api_key = openai_key
        elif not "OPENAI_API_KEY" in os.environ:
            raise Exception("No openai key set!")

        conv = {"gpt3": "gpt-3.5-turbo", "gpt-3": "gpt-3.5-turbo", "chatgpt": "gpt-3.5-turbo", "gpt4": "gpt-4",
                "gpt-16k": "gpt-3.5-turbo-16k"}
        self.model = conv.get(model_path_or_name, model_path_or_name)
        self.symbols["ASSISTANT"] = "assistant"
        self.symbols["USER"] = "user"
        self.symbols["SYSTEM"] = "system"
        self.finish_meta = {}

    # @abstractmethod
    def tokenize(self, text):
        encoding = tiktoken.encoding_for_model(self.model)
        return encoding.encode(text)

    def tokenize_as_str(self, text):
        encoding = tiktoken.encoding_for_model(self.model)
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
            # self.test_txt += token
            yield token, None

    def create_native_completion(self, text, max_tokens=256, stop=None, keep_dict=False, token_prob_delta=None,
                                 token_prob_abs=None,
                                 log_probs=None, **kwargs):
        if isinstance(text, str):
            raise Exception("Native OpenAI call only supports calls via a json chat format")
        if token_prob_abs:
            raise Exception("OpenAI API only supports relative logit chance change")
        if log_probs:
            raise Exception("OpenAI API does not support retrieval of logits")
        start = timer()
        if token_prob_delta:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=text,
                logit_bias=token_prob_delta,
                stop=stop,
                **kwargs
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=text,
                stop=stop,
                **kwargs
            )
        response_txt = response["choices"][0]["message"]["content"]
        end = timer()

        self.finish_meta["finish_reason"] = response["choices"][0]["finish_reason"]
        tok_in = response["usage"]["prompt_tokens"]
        tok_gen = response["usage"]["completion_tokens"]
        tok_total = response["usage"]["total_tokens"]

        self.finish_meta["tokens"] = {"prompt_tokens": tok_in, "generated_tokens": tok_gen, "total_tokens": tok_total}
        self.finish_meta["timings"] = {"total_time": round(end - start,3)}
        self.finish_meta["t_per_s"] = {"token_total_per_s": (tok_total) / (end - start)}

        cost_in = OpenAI.pricing[self.model]["input"] * tok_in / OpenAI.pricing_meta["token_unit"]
        cost_out = OpenAI.pricing[self.model]["output"] * tok_gen / OpenAI.pricing_meta["token_unit"]
        self.finish_meta["cost"] = {"input": round(cost_in, 3), "output": round(cost_out, 5),
                                    "total": round(cost_out + cost_in, 5),
                                    "total_cent": round((cost_out + cost_in) * 100, 3),
                                    "unit": OpenAI.pricing_meta["currency"]}

        if keep_dict:
            return response
        return response_txt

    def create_native_generator(self, text, keep_dict=False, token_prob_delta=None,
                                token_prob_abs=None, **kwargs):
        if token_prob_abs:
            raise Exception("OpenAI API only supports relative logit chance change")

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=text,
            stream=True,
            logit_bias=token_prob_delta,
            # stop_sequences=stop,
            **kwargs
        )

        if keep_dict:
            return response
        else:
            return self._extract_message_from_generator(response)

    def build_prompt(self, conv_history=None, system_msg=None):
        if not conv_history:
            conv_history = self.conv_history
        if not system_msg:
            system_msg = self.system_msg
        prompt = []
        if "content" in system_msg:
            prompt.append({"role": self.symbols["SYSTEM"], "content":
                self._replace_symbols(system_msg["content"])})
        for i in conv_history:
            prompt.append({"role": self.symbols[str(i["role"])], "content": self._replace_symbols(i["content"], i)})
        return prompt
