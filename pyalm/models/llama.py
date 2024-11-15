from pyalm.internal.alm import ALM
import os
from timeit import default_timer as timer

import llama_cpp


class LLaMa(ALM):

    def __init__(self, model_path_or_name, verbose=0, no_system_msg_supported= False,**kwargs):
        super().__init__(model_path_or_name, verbose=verbose, **kwargs)
        self.model_name = model_path_or_name.split("/")[-1]
        self.model = llama_cpp.Llama(model_path_or_name, verbose=verbose, **kwargs)

        llama_specifics = {"ASSISTANT": "assistant", "USER": "user", "SYSTEM": "system"}
        self._built_in_symbols.update(llama_specifics)
        self.no_system_msg_supported = no_system_msg_supported



    def create_native_completion(self, text, temp=0, **kwargs):
        if isinstance(text, str):
            return self.model.create_completion(text, temperature=temp, **kwargs)
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
            if self.no_system_msg_supported:
                prompt.insert(0, {"role": "user", "content": "###System message###\n"+system_msg})
            else:
                prompt.insert(0, {"role": self.symbols["SYSTEM"], "content": system_msg})

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
        return prompt
