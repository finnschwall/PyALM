from pyalm.internal.alm import ALM
from openai import OpenAI as _OpenAI, AzureOpenAI as _AzureOpenAI
import os
import tiktoken
from timeit import default_timer as timer


class OpenAI(ALM):
    available_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]
    """All models that are known to work"""
    pricing = {"gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
               "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
               "gpt-4": {"input": 0.03, "output": 0.06}, }
    """Pricing per input and output tokens in a prompt. Streaming costs the same"""
    pricing_meta = {"currency": "$", "token_unit": 1000}


    def get_available_models(self):
        return self.client.models.list().data

    def __init__(self, model_path_or_name, openai_key=None, verbose=0, azure_endpoint=None, api_version= "2023-05-15",
                 **kwargs):
        """
        :param model_path_or_name: Model name. Must be one of openais available models
        :param openai_key: OpenAI API key. Can also be set as environment variable OPENAI_API_KEY
        :param azure_endpoint: Connect to Azure server instead of OpenAI.
        :param api_version: Azure API version
        """
        super().__init__(model_path_or_name, verbose=verbose)
        conv = {"gpt3": "gpt-3.5-turbo", "gpt-3": "gpt-3.5-turbo", "chatgpt": "gpt-3.5-turbo", "gpt4": "gpt-4",
                "gpt-16k": "gpt-3.5-turbo-16k"}
        self.model = conv.get(model_path_or_name, model_path_or_name)

        if azure_endpoint:
            if openai_key:
                self.client = _AzureOpenAI(
                    api_key=openai_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint
                )
            elif "OPENAI_API_KEY" in os.environ:
                self.client = _AzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=azure_endpoint
                )
            else:
                raise Exception("No openai key set!")
        else:
            if openai_key:
                self.client = _OpenAI(api_key=openai_key)
            elif "OPENAI_API_KEY" in os.environ:
                self.client = _OpenAI()
            else:
                raise Exception("No openai key set!")

        openai_specifics = {"ASSISTANT": "assistant", "USER": "user", "SYSTEM": "system"}
        self._built_in_symbols.update(openai_specifics)
        self.settings.prompt_obj_is_str = False
        self.total_tokens = 0

    # @abstractmethod
    def tokenize(self, text):
        encoding = tiktoken.encoding_for_model(self.model)
        return encoding.encode(text)

    def tokenize_as_str(self, text):
        encoding = tiktoken.encoding_for_model(self.model)
        encoded = encoding.encode(text)
        return [encoding.decode_single_token_bytes(token).decode("utf-8") for token in encoded]

    # TODO add static variant
    def get_n_tokens(self, text):
        return len(self.tokenize(text))

    def _extract_message_from_generator(self, gen):

        for i in gen:
            try:

                token = i.choices[0].delta.content
                finish_reason = i.choices[0].finish_reason
                if finish_reason:
                    self.finish_meta["finish_reason"] = finish_reason
                if token is None:
                    continue
                    # break
                yield token, None
            except Exception as e:
                print(e)
                pass
            # self.test_txt += token


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
            response = self.client.chat.completions.create(model=self.model,
            messages=text,
            logit_bias=token_prob_delta,
            stop=stop,
            **kwargs)
        else:
            response = self.client.chat.completions.create(model=self.model,
            messages=text,
            stop=stop,
            **kwargs
            )
        response_txt = response.choices[0].message.content
        end = timer()

        self.finish_meta["finish_reason"] = response.choices[0].finish_reason
        tok_in = response.usage.prompt_tokens
        tok_gen = response.usage.completion_tokens
        tok_total = response.usage.total_tokens

        self.finish_meta["tokens"] = {"prompt_tokens": tok_in, "generated_tokens": tok_gen, "total_tokens": tok_total}
        self.total_tokens += tok_total
        self.finish_meta["timings"] = {"total_time": round(end - start,2)}
        self.finish_meta["t_per_s"] = {"token_total_per_s": round((tok_total) / (end - start),2)}

        # try:
        #     cost_in = OpenAI.pricing[self.model]["input"] * tok_in / OpenAI.pricing_meta["token_unit"]
        #     cost_out = OpenAI.pricing[self.model]["output"] * tok_gen / OpenAI.pricing_meta["token_unit"]
        #     self.finish_meta["cost"] = {"input": round(cost_in, 3), "output": round(cost_out, 5),
        #                                 "total": round(cost_out + cost_in, 5),
        #                                 "total_cent": round((cost_out + cost_in) * 100, 3),
        #                                 "unit": OpenAI.pricing_meta["currency"]}
        # except:
        #     pass

        if keep_dict:
            return response
        return response_txt

    def create_native_generator(self, text, keep_dict=False, token_prob_delta=None,
                                token_prob_abs=None, **kwargs):
        if token_prob_abs:
            raise Exception("OpenAI API only supports relative logit chance change")
        if token_prob_delta:
            response = self.client.chat.completions.create(model=self.model,
            messages=text,
            stream=True,
            logit_bias=token_prob_delta,
            # stop_sequences=stop,
            **kwargs)
        else:
            response = self.client.chat.completions.create(model=self.model,
            messages=text,
            stream=True,
            **kwargs
            )
        if keep_dict:
            return response
        else:
            return self._extract_message_from_generator(response)

    def build_prompt(self, conv_history=None, system_msg=None):
        if not conv_history:
            conv_history = self.conversation_history.tracker
        if not system_msg:

            if self.conversation_history.system_message:
                system_msg = self.conversation_history.system_message
            else:
                system_msg = self.replace_symbols(self.replace_symbols(self.system_msg_template))
        if system_msg and len(system_msg) <3:
            system_msg = None
        prompt = []

        if system_msg and system_msg != "":
            system_msg = self.replace_symbols(system_msg)
            if "CONTEXT" in self.symbols and self.symbols["CONTEXT"]:
                system_msg += "\n\nGATHERED CONTEXT/KNOWLEDGE:\n" + self.symbols["CONTEXT"]
            prompt.insert(0, {"role": self.symbols["SYSTEM"], "content": system_msg
                })


        for i in conv_history:
            prompt_content = ""
            if "content" in i and i["content"]:
                prompt_content = self.replace_symbols(i["content"], i)
            if "code" in i and i["code"]:
                code_str = f"\n{self.settings.function_sequence[0]}\n"+i["code"]+f"\n{self.settings.function_sequence[1]}"
                if "return_value" in i and i["return_value"]:
                    code_str += "\nRETURN:\n"#+i["return_value"]
                    if isinstance(i["return_value"], str):
                        code_str += i["return_value"]
                    else:
                        try:
                            code_str += str(i["return_value"])
                        except:
                            code_str += "RETURN VALUE OF FUNCTION IS NOT CONVERTIBLE TO STRING!"
                else:
                    code_str += "\nRETURN:\nNone(OK)"
                prompt_content += code_str
            if prompt_content:
                prompt.append({"role": self.symbols[str(i["role"])], "content": prompt_content})
            # if "code" in i and i["code"]:
            #     code_str = "CODE_START\n"+i["code"]+"\nCODE_START"
            #     if "return_value" in i and i["return_value"]:
            #         code_str += "\nRETURN:\n"+i["return_value"]
            #     else:
            #         code_str += "\nRETURN:\nNone"
            #     prompt.append({"role": self.symbols[str(i["role"])], "content":code_str})
            # else:
            #     prompt.append({"role": self.symbols[str(i["role"])], "content": self.replace_symbols(i["content"], i)})
        # if "CONTEXT" in self.symbols and self.symbols["CONTEXT"]:
        #     last_usr_entry, depth = self.conversation_history.get_last_message(ConversationRoles.USER, True)
        #     if depth == 0 and last_usr_entry:
        #         prompt[-1]["content"] = "#CONTEXT_START\n"+ self.symbols["CONTEXT"]+"\n#CONTEXT_END\n" + prompt[-1]["content"]
        return prompt
