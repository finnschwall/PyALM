from pyalm.internal.alm import ALM
import os
from aleph_alpha_client import Client, CompletionRequest, Prompt, SummarizationRequest, Document, Image
import requests
import json
from timeit import default_timer as timer


class AlephAlpha(ALM):
    available_models = ['luminous-supreme', 'luminous-base', 'luminous-extended-control', 'luminous-base-control',
                        'luminous-supreme-control', 'luminous-extended']
    pricing = {"luminous-bas": 0.03, "luminous-extended": 0.045, "luminous-supreme": 0.175, "luminous-base-control": 0.0375, "luminous-extended-control": 0.05625,
               "luminous-supreme-control": 0.21875}
    """Pricing per token"""
    pricing_img = {"luminous-base": 0.03024, "luminous-extended": 0.04536}
    """Cost per processed image"""
    pricing_factors = {"Complete": {"input": 1, "output": 1.1}, "Summarize": {"input": 1.3, "output": 1.1}}
    """Pricing factor depending on model and whether it is prompt or output"""
    pricing_meta = {"currency": "credits", "token_unit": 1000, "€/Credits": 0.2}

    def __init__(self, model_path_or_name, aleph_alpha_key=None, verbose=0, n_ctx=2048, **kwargs):
        super().__init__(model_path_or_name, verbose=verbose)
        if aleph_alpha_key:
            self.api_key = aleph_alpha_key
        elif "AA_TOKEN" in os.environ:
            self.api_key = os.getenv("AA_TOKEN")
        elif not "AA_TOKEN" in os.environ:
            raise Exception("No aleph_alpha_key key set!")
        self.model = model_path_or_name
        self.client = Client(token=self.api_key)
        self.llm = self.client
        self.tokenizer = self.client.tokenizer(self.model)
        self.finish_meta = {}
        self.prompt_text_is_str = True
        self.pricing = {"gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}}

    def detokenize(self, toks):
        if not isinstance(toks, list):
            toks = [toks]
        return self.tokenizer.decode(toks)

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def tokenize_as_str(self, text):
        encoded = self.tokenize(text)
        return [self.detokenize(token).decode("utf-8") for token in encoded]

    def get_n_tokens(self, text):
        return len(self.tokenize(text))

    def get_remaining_credits(self):
        """
        How many credits are still available in the given API key

        :return: remaining credits
        """
        url = "https://api.aleph-alpha.com/users/me"

        payload = {}
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer ' + self.api_key
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        dic = json.loads(response.text)
        return dic["credits_remaining"]

    def create_native_completion(self, text, max_tokens=256, stop=None, token_prob_delta=None,
                                 token_prob_abs=None, log_probs=None, *, keep_dict=False, **kwargs):
        start = timer()
        input_tokens = self.get_n_tokens(text)
        if token_prob_abs:
            raise Exception("Aleph alpha only supports relative logit chance change")
        if token_prob_delta:
            request = CompletionRequest(
                prompt=Prompt.from_text(text),
                maximum_tokens=max_tokens,
                logit_bias=token_prob_delta,
                log_probs=log_probs,
                stop_sequences=stop,
                **kwargs
            )
        else:
            request = CompletionRequest(
                prompt=Prompt.from_text(text),
                maximum_tokens=max_tokens,
                log_probs=log_probs,
                stop_sequences=stop,
                **kwargs
            )
        response = self.client.complete(request, model=self.model)
        response_txt = response.completions[0].completion
        output_tokens = self.get_n_tokens(response_txt)
        end = timer()

        self.finish_meta["finish_reason"] = response.completions[0].finish_reason
        self.finish_meta["tokens"] = {"prompt_tokens": input_tokens, "generated_tokens": output_tokens,
                                      "total_tokens": input_tokens + output_tokens}
        self.finish_meta["timings"] = {"total_time": round(end - start, 3)}
        self.finish_meta["t_per_s"] = {"token_total_per_s": round((input_tokens + output_tokens) / (end - start), 3)}

        cost_in = AlephAlpha.pricing[self.model] * input_tokens / AlephAlpha.pricing_meta["token_unit"]*AlephAlpha.pricing_factors["Complete"]["input"]
        cost_out = AlephAlpha.pricing[self.model] * output_tokens / AlephAlpha.pricing_meta["token_unit"]*AlephAlpha.pricing_factors["Complete"]["output"]
        self.finish_meta["cost"] = {"input": round(cost_in, 3), "output": round(cost_out, 5),
                                    "total": round(cost_out + cost_in, 5),
                                    "unit": AlephAlpha.pricing_meta["currency"],
                                    "total_euro": round((cost_out + cost_in)*AlephAlpha.pricing_meta["€/Credits"],5)
                                    }

        logprobs = response.completions[0].log_probs

        if keep_dict:
            return response
        if logprobs:
            return response, logprobs
        return response_txt

    def create_native_generator(self, text, keep_dict=False, token_prob_delta=None,
                                token_prob_abs=None, max_tokens=256, **kwargs):
        raise Exception("Aleph alpha does not support streaming")

    def build_prompt(self, preserve_flow=False):
        return self.build_prompt_as_str(1, 0, block_gen_prefix=preserve_flow)

    def summarize(self, *, text= None, path_to_docx=None):
        """
        Summarize a text using the current model

        :param text: Text to summarize
        :param path_to_docx: Alternative to text. Summarize a .docx document
        :return: summarized text as string
        """
        if text:
            request = SummarizationRequest(document=Document.from_text(text))
        elif path_to_docx:
            request = SummarizationRequest(document=Document.from_docx_file(path_to_docx))
        else:
            raise Exception("Nothing given for summarization")
        response = self.client.summarize(request=request)
        summary = response.summary
        return summary

    @staticmethod
    def image_from_source(source):
        """
        Create Aleph compatible image from e.g. file, url etc.

        :param source:
        :return: Aleph compatible image obj
        """
        return Image.from_image_source(image_path)

    def multimodal_completion(self, prompt_list, max_tokens = 256, stop=None,**kwargs):
        """
        Prompt the model using multimodal input

        :param prompt_list: A list of texts and images.
        :param max_tokens: Max tokens to return
        :param stop: List of strings to stop at
        :param kwargs: kwargs
        :return: Text
        """
        prompt = Prompt(items=prompt_list)
        if stop:
            request = CompletionRequest(prompt=prompt, maximum_tokens=max_tokens, stop_sequences=stop, **kwargs)
        else:
            request = CompletionRequest(prompt=prompt, maximum_tokens=max_tokens,**kwargs)
        response = client.complete(request=request, model=self.model)
        return response.completions[0].completion


