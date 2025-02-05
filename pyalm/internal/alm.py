import datetime

import os
import re
from timeit import default_timer as timer
from typing import Type
from warnings import warn
import contextlib
import io
from abc import abstractmethod

import numpy as np

import rixaplugin
from pyalm.chat import system_msg_templates
from pyalm.internal.state import *

import logging
alm_logger = logging.getLogger("rixa.alm")

class FunctionFormat(enum.Enum):
    PYDOC = "PYDOC"
    JSON = "JSON"
    MODEL_SPECIFIC = "MODEL_SPECIFIC"

    def __str__(self) -> str:
        return self.value


def change_latex_delimiters(latex_str):
    modified_str = latex_str.replace("\\[", "$$").replace("\\]", "$$")
    modified_str = modified_str.replace("\\(", "$").replace("\\)", "$")
    return modified_str


class ParseStatus(enum.Enum):
    """
    Used to classify outcomes when trying to handle LLMs function call
    """
    UNDEFINED = "UNDEFINED"
    NO_FUNC_SEQUENCE_FOUND = "NO_FUNC_SEQUENCE_FOUND"
    UNPARSEABLE_FUNC_FOUND = "UNPARSEABLE_FUNC_FOUND"
    PARSED_DICT_RETURN = "PARSED_DICT_RETURN"
    PARSED_EXECUTED_OK = "PARSED_EXECUTED_OK"
    PARSED_EXECUTED_ERR = "PARSED_EXECUTED_ERR"

    def __str__(self) -> str:
        return self.value


class Symbols(dict):
    def __setitem__(self, key, value):
        key = key.upper()
        if key == "FUNCTION_START" or key == "FUNCTION_END":
            raise KeyError(f"'{key}' is a reference. Change 'function_sequence' tuple in settings")
        if key == "FUNCTION_CALL":
            raise KeyError(f"'{key}' is a reference. Change 'function_integration_template' in settings")

        super().__setitem__(key, value)


# TODO move verbose into settings
# TODO check if tutorial works
class ALM:
    """
    Base class. Don't instantiate on its own
    """

    @property
    def verbose(self):
        return self.settings.verbose

    @verbose.setter
    def verbose(self, v_new):
        self.settings.verbose = v_new

    @property
    def system_msg(self):
        return self.conversation_history.system_message

    @system_msg.setter
    def system_msg(self, system_msg):
        self.conversation_history.system_message = system_msg

    @property
    def symbols(self):
        symbols = dict(self._built_in_symbols, **self.user_symbols)
        # symbols.update(self._temp_symbols)
        return symbols

    @symbols.setter
    def symbols(self, system_msg):
        raise Exception("Add or modify symbols via 'user_symbols'")

    @property
    def preserved_sequences(self):
        return self.settings.preserved_sequences

    @preserved_sequences.setter
    def preserved_sequences(self, value):
        self.settings.preserved_sequences = value

    def __init__(self, model_path_or_name, verbose=0, enable_functions=False, no_system_msg_supported= False,**kwargs):
        self.no_system_msg_supported = no_system_msg_supported
        self.code_calls = None
        if enable_functions:
            self.enable_automatic_function_calls()
        self.model = model_path_or_name
        self.model_name = str(model_path_or_name)

        self.settings = ALMSettings(verbose=verbose)
        self.settings.verbose = verbose
        self.model_meta = {"model_name": model_path_or_name}

        self.conversation_history = ConversationTracker()

        self.include_context_msg = True
        self.include_function_msg = True
        self.code_call_sys_msg = system_msg_templates.function_call_msg
        self.context_sys_msg = system_msg_templates.context_msg
        self.context_with_code_sys_msg = system_msg_templates.context_with_code_msg
        self.system_msg_template = "SCENARIO SPECIFIC INSTRUCTIONS:\n[[USR_SYSTEM_MSG]]\n\n\n[[CODE_CALL_SYSTEM_MSG]]\n\n\n[[CONTEXT_SYSTEM_MSG]]"

        self.code_callback = rixaplugin.execute_code
        """
        Function that will be called to execute code
        """

        self.n_ctx = -1 if not kwargs.get("n_ctx") else kwargs["n_ctx"]

        def _include_code_call_sys_msg(match, symbols, text=None):
            if self.include_function_msg and self.symbols.get("LIST_OF_FUNCTIONS") and self.symbols[
                "LIST_OF_FUNCTIONS"] != "":
                return self.code_call_sys_msg
            else:
                return ""

        def _include_context_sys_msg(match, symbols, text=None):
            if self.include_context_msg and self.symbols.get("CONTEXT") and self.symbols["CONTEXT"] != "":
                if self.include_function_msg and self.symbols.get("LIST_OF_FUNCTIONS") and self.symbols[
                    "LIST_OF_FUNCTIONS"] != "":
                    return self.context_with_code_sys_msg
                else:
                    return self.context_sys_msg
            else:
                return ""

        self._built_in_symbols = {
            "FUNCTION_START": lambda match, symbols, text=None: self.settings.function_sequence[0],
            "FUNCTION_END": lambda match, symbols, text=None: self.settings.function_sequence[1],
            "ASSISTANT": "assistant", "USER": "user", "SYSTEM": "system",
            "TO_USER": self.settings.to_user_sequence,
            "FUNCTION_CALL": lambda match, symbols, text=None: self.replace_symbols(
                self.settings.function_integration_template, additional_symbols=symbols),
            "CODE_CALL_SYSTEM_MSG": _include_code_call_sys_msg,
            "CONTEXT_SYSTEM_MSG": _include_context_sys_msg,
            "USR_SYSTEM_MSG": "",
        "DATE": lambda match, symbols, text=None: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),}

        self.user_symbols = Symbols()
        """
        Variable symbols that will get replaced when building prompt. Either string or function pointer 
        """

        # TODO modify PyLoT so that this works with doc, signature etc. included
        self.available_functions = {}
        """Function names and the callable functions available for the model"""

        self.raw_generated_text = ""

        self._finish_meta_template = {"finish_reason": "Unknown", "timings": {}, }
        self.finish_meta = dict(self._finish_meta_template)

        self.jupyter_gui = False

    def enable_automatic_function_calls(self):
        self.settings.global_enable_function_calls = True
        self.set_system_message(self.system_msg, prepend_function_support=True)

    # TODO implement
    def pop_entry(self):
        """
        Remove last element from conversation tracker. Automatically takes care of e.g. split messages due to function calls.
        :return: The popped entry
        """
        tracker = self.conversation_history.tracker
        if len(tracker) == 0:
            raise Exception("No items to pop")
        if len(tracker) == 1:
            return tracker.pop()
        last_role = tracker[-1]["role"]
        index = -1
        for i in range(len(tracker) - 1, -1, -1):
            if self.conversation_history.inversion_scheme.get(tracker[i]["role"]) == last_role:
                index = i
                break
        if index == -1:
            temp = tracker
            self.conversation_history.tracker = []
            return temp
        ret = []
        for i in range(len(tracker) - 1, index, -1):
            ret.append(tracker.pop(i))
        pass

    def adopt_from_alm(self, other: Type['ALM']):
        """
        Copy state from other ALM into this. This is not a deep copy!

        :param other: Other ALM
        """
        # self.conv_history = other.conv_history
        self.system_msg = other.system_msg
        self.verbose = other.verbose
        self.settings.update(other.settings)

    def _repl(self, match, symbols, text=None):
        """
        Callable for re.sub in _replace_symbols

        :param match: regex match
        :param text: whole text
        :param temp_symbols: additional symbols
        :return: replacement
        """
        if match[1] in symbols:
            val = symbols[match[1]]
            if isinstance(val, str):
                return symbols[match[1]]
            else:
                try:
                    return val(match, symbols, text)
                except Exception as e:
                    raise Exception("An error occurred while trying to substitute symbols for prompt:\n" + str(e))
        return f"#SYMBOL_MISSING: {match[1]}#"

    # def _build_func_template(self, match, symbols, text):
    #     return self._replace_symbols(self.settings["function_integration_template"], temp_symbols=temp_symbols)

    def replace_symbols(self, text, entry=None, additional_symbols=None):
        """
        Replace symbols in a conv history entry or text

        :param text: text with symbols in it
        :param entry: optional, conv history entry to use for replacement
        :param temp_symbols: additional symbols to be replaced
        :return: text with substitutions
        """
        symbols = dict(self._built_in_symbols, **self.symbols)
        if additional_symbols:
            symbols.update(additional_symbols)
        if entry:
            if "function_calls" in entry:
                if "original_call" in entry["function_calls"]:
                    symbols["FUNCTION_SEQUENCE"] = entry["function_calls"]["original_call"]
                if "return" in entry["function_calls"]:
                    symbols["FUNCTION_RETURN_VALUE"] = entry["function_calls"]["return"]
        pattern = r'\[\[(.*?)\]\]'
        text = re.sub(pattern, lambda match: self._repl(match, symbols, text), text)
        return text

    # TODO fix
    def save_state(self, path=None):
        """
        Saves the ALMs entire state (excluding the model itself)

        :param path: Where to save. If not specified string is returned
        :return: None or state as yaml
        """
        raise NotImplementedError
        state = {"system_msg": self.system_msg, "conv_history": self.conv_history, "settings": self.settings,
                 "symbols": self.symbols, "preserved_sequences": self.preserved_sequences}
        yaml_str = yaml.dump(state, sort_keys=False)
        if not path:
            return yaml_str

        with open(path, "w") as f:
            f.write(yaml_str)

    # TODO implement string return
    # TODO FIX
    def save_history(self, path=None):
        """
        Saves the ALMs conversation history

        :param path: Where to save. If not specified string is returned
        :return: None or history as yaml like string
        """
        self.conversation_history.save_to_yaml(path)

    # TODO fix
    def load_history(self, path_or_text, is_file=True):
        """
        Loads a conversation history

        :param path_or_text: Either path to a file or a yaml like string
        """
        self.conversation_history = ConversationTracker.load_from_yaml(path_or_text, is_file)

    def set_system_message(self, msg="", prepend_function_support=False):
        """
        Change the system message

        :param msg: new message
        :param prepend_function_support: Automatically change message to include function calling
        """
        if prepend_function_support:
            msg = self.settings.function_inclusion_instruction_system_msg + msg
        self.system_msg = msg

    @abstractmethod
    def tokenize(self, text):
        """
        Text to token as vector representation

        :param text:
        :return: List of tokens as ints
        """
        raise NotImplementedError()

    @abstractmethod
    def tokenize_as_str(self, text):
        """
        Text to token as vector representation but each token is converted to string

        :param text:
        :return: List of tokens as strings
        """
        raise NotImplementedError()

    @abstractmethod
    def get_n_tokens(self, text):
        """
        How many tokens are in a string

        :param text: tokenizable text
        :return: amount
        """
        raise NotImplementedError()

    @abstractmethod
    def create_native_generator(self, text, keep_dict=False, token_prob_delta=None,
                                token_prob_abs=None, **kwargs):
        """
        Library native generator for tokens. Different for each library. No processing of output is done

        :param text: Prompt or prompt obj
        :param keep_dict: If library or API returns something else than raw tokens, whether to return native format
        :param token_prob_delta: dict, Absolute logits for tokens
        :param token_prob_abs: dict, relative added number for token logits
        :param kwargs: kwargs
        :return: generator
        """
        raise NotImplementedError()

    @abstractmethod
    def create_native_completion(self, text, max_tokens=256, stop=None, keep_dict=False, token_prob_delta=None,
                                 token_prob_abs=None,
                                 log_probs=None, **kwargs):
        """
        Library native completion retriever. Different for each library. No processing of output is done

        :param text: Prompt or prompt obj
        :param max_tokens: maximum tokens generated in completion
        :param stop: Additional stop sequences
        :param keep_dict: If library or API returns something else than raw tokens, whether to return native format
        :param token_prob_delta: dict, relative added number for token logits
        :param token_prob_abs: dict, Absolute logits for tokens
        :param log_probs: int, when not None return the top X log probs and their tokens
        :param kwargs: kwargs
        :return: completion
        """
        raise NotImplementedError()

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
                code_str = f"\n{self.settings.function_sequence[0]}\n"+i["code"]#+f"\n{self.settings.function_sequence[1]}"
                if "return_value" in i and i["return_value"]:
                    code_str += "\n#--RETURN FROM CODECALL--\n"
                    if isinstance(i["return_value"], str):
                        code_str += "#"+i["return_value"].replace("\n", "\n#")
                    else:
                        try:
                            code_str += str(i["return_value"])
                        except:
                            code_str += "RETURN VALUE OF FUNCTION IS NOT CONVERTIBLE TO STRING!"
                else:
                    if not self.settings.to_user_sequence in code_str:
                        code_str += "\nRETURN:\nNone"
                if not self.settings.to_user_sequence in code_str:
                    code_str += f"\n{self.settings.function_sequence[1]}"
                prompt_content += code_str
            if prompt_content:
                prompt.append({"role": i["role"], "content": prompt_content})
        return prompt

    # @abstractmethod
    # def build_prompt(self, preserve_flow=False):
    #     """
    #     Build prompt in format native to library
    #
    #     :param preserve_flow: Block suffix for purely text based models
    #     :return: prompt obj
    #     """
    #     raise NotImplementedError()

    def reset_tracker(self, purge=False):
        """
        Remove all tracker entries

        :param purge: Complete reset including e.g. system message
        """
        if purge:
            self.conversation_history = ConversationTracker()
        else:
            self.conversation_history.reset_tracker()

    def add_tracker_entry(self, *args, **kwargs):
        """
        Add a new entry to the tracker. More info is in
        https://github.com/finnschwall/PyALM/blob/main/format_specifications.md
        """
        self.conversation_history.add_entry(*args, **kwargs)

    def create_completion(self, text_obj=None, verbose=None, enable_function_calls=None, chat=True,
                          token_prob_delta=None, token_prob_abs=None, handle_functions=None, stop=None, **kwargs):
        """
        Create completion with automatic prompt building, function calling etc.

        :param text: Prompt or prompt obj
        :param max_tokens: maximum tokens generated in completion
        :param stop: Additional stop sequences
        :param chat: Only applicable for pure text models. Will pass just text to model without building a history.
        :param keep_dict: If library or API returns something else than raw tokens, whether to return native format
        :param token_prob_delta: dict, relative added number for token logits
        :param token_prob_abs: dict, Absolute logits for tokens
        :param log_probs: int, when not None return the top X log probs and their tokens
        :param handle_functions: Whether to call functions or return list with attemptedcalls
        :param kwargs: kwargs
        :return: completion
        """
        self.finish_meta = dict(self._finish_meta_template)
        self.raw_generated_text = ""
        if not self.settings.prompt_obj_is_str:
            chat = True
        if enable_function_calls is None:
            enable_function_calls = self.settings.global_enable_function_calls
        if handle_functions is None:
            handle_functions = enable_function_calls

        start = timer()
        if not verbose:
            verbose = self.verbose

        if not stop:
            stop = []

        if isinstance(stop, str):
            stop = [stop]

        if enable_function_calls:
            if not chat:
                warn("Enabling function calls in non-chat scenario can lead to strange results.")
            stop.append(self.settings.function_sequence[1])

        add_kwargs = {"stop": stop, "token_prob_delta": token_prob_delta, "token_prob_abs": token_prob_abs}

        if chat:
            if text_obj:
                self.add_tracker_entry(text_obj, ConversationRoles.USER)
            prompt_obj = self.build_prompt()
            # print(self.build_prompt_as_str(use_build_prompt=True))
            self.prompt = prompt_obj
            if self.settings.prompt_obj_is_str and self.settings.include_conv_id_as_stop:
                stop.append(self.symbols["USER"])
                stop.append(self.symbols["ASSISTANT"])
            try:
                ret_text = self.create_native_completion(prompt_obj, **add_kwargs,
                                                         **kwargs)  # return_factory(prompt_obj)
            except Exception as e:
                self.pop_entry()
                raise e
        elif text_obj and self.settings.prompt_obj_is_str:
            try:
                ret_text = self.create_native_completion(text_obj, **add_kwargs, **kwargs)
            except Exception as e:
                self.pop_entry()
                raise e
        if not chat and not text_obj:
            raise Exception("No prompt given!")

        self.raw_generated_text += ret_text
        if not enable_function_calls:
            if chat:
                self.add_tracker_entry(ret_text, ConversationRoles.ASSISTANT)
            end = timer()
            self.finish_meta["total_finish_time"] = end - start
            return self.raw_generated_text

        try:
            stop.remove(self.settings.function_sequence[1])
        except:
            pass
        prompt_obj = self.build_prompt()
        ret_text = self.create_native_completion(prompt_obj, **add_kwargs, **kwargs)

        self.raw_generated_text += ret_text

        if chat:
            self.add_tracker_entry(ret_text, ConversationRoles.ASSISTANT, )
        end = timer()
        self.finish_meta["total_finish_time"] = end - start
        return self.raw_generated_text

    def create_completion_plugin(self, conv_tracker=None, context=None, func_list=None, system_msg=None, code_calls=0,
                                 username=None, temp=None, metadata=None):
        import rixaplugin.sync_api as api

        self.finish_meta = dict(self._finish_meta_template)
        if not metadata:
            metadata = {"total_tokens":0}
        self.raw_generated_text = ""
        start = timer()
        if conv_tracker:
            self.conversation_history = conv_tracker
            self.conversation_history.metadata["model_name"] = self.model_name


        self.user_symbols["LIST_OF_FUNCTIONS"] = ""
        self.user_symbols["CONTEXT"] = ""
        self.user_symbols["USR_SYSTEM_MSG"] = "" if not system_msg else system_msg

        if func_list:
            self.user_symbols["LIST_OF_FUNCTIONS"] = func_list

        if context:
            self.user_symbols["CONTEXT"] = context
        prompt_obj = self.build_prompt()
        if isinstance(prompt_obj[0], dict):
            len_system = len(prompt_obj[0]["content"])
            conv_sizes = [len_system]+[len(x["content"]) for x in prompt_obj[1:][::-1]]
            # when 60% of n_ctx is used, remove oldest entries until we are below again
            cumsum = np.cumsum(conv_sizes)
            max_chars = self.n_ctx*0.6*3
            # factor 4 is because 1 token is ~4 characters
            # factor 0.6 to make sure the model does not run out of context even for long answers
            if self.n_ctx != -1 and cumsum[-1] > max_chars:
                api.show_message("Chat has been truncated due to being too long. Consider deleting the chat history")
                idx = np.where(cumsum > max_chars)[0]
                if len(idx) == 0:
                    idx = -1
                else:
                    idx = -idx[0]
                    if idx == 0:
                        idx = -1
                prompt_obj = [prompt_obj[0]]+prompt_obj[idx:]
        self.prompt = prompt_obj

        if conv_tracker:
            with open(rixaplugin.settings.WORKING_DIRECTORY + "/chat_tmp.txt", "w") as f:
                text = f"##########\nCalling model\nUser: {username}\nCurrent complete prompt:\n##########\n"
                text += self.build_prompt_as_str(use_build_prompt=True, include_system_msg=True)
                f.write(text)
        try:
            with open(rixaplugin.settings.WORKING_DIRECTORY + "/chat_tmp.txt", "a") as f:
                text = f"\n\n\n##########\nNEW PROMPT:\n##########\n{self.build_prompt_as_str(use_build_prompt=True,include_system_msg=False)}\n"
                f.write(text)
            if temp:
                ret_text = self.create_native_completion(prompt_obj, temp=temp)
            else:
                ret_text = self.create_native_completion(prompt_obj)
            with open(rixaplugin.settings.WORKING_DIRECTORY + "/chat_tmp.txt", "a") as f:
                f.write("\n\n\n##########\nRAW RETURN FROM MODEL:\n##########\n" + ret_text + "\n##########\n\n")
        except Exception as e:
            self.pop_entry()
            raise e
        metadata["total_tokens"] += self.finish_meta["tokens"]["total_tokens"]

        start_seq = self.settings.function_sequence[0]
        end_seq = self.settings.function_sequence[1]
        pattern = re.escape(start_seq) + '(.*?)' + re.escape(end_seq)
        matches = [(m.group(1), m.span()) for m in re.finditer(pattern, ret_text, re.DOTALL)]
        if len(matches) == 0:
            ret_text_copy = ret_text + end_seq
            pattern = re.escape(start_seq) + '(.*?)' + re.escape(end_seq)
            matches = [(m.group(1), m.span(), m) for m in re.finditer(pattern, ret_text_copy, re.DOTALL)]
            if len(matches) == 0:
                self.conversation_history.add_entry(change_latex_delimiters(ret_text), ConversationRoles.ASSISTANT)
                return self.conversation_history, metadata
        func_seq = matches[0][0]
        code_calls += 1
        if code_calls >= 3:
            self.conversation_history.add_entry(
                "RIXA failed while trying to call a function due to exceeding call limit.",
                role=ConversationRoles.ASSISTANT
                )
            return self.conversation_history, metadata
        try:
            func_seq_truncated = func_seq.strip()
            # sometimes the model misses the end sequence and tries to end with #TO_USER. So we see if the end sequence is missing
            # if thats the case AND #TO_USER is in the function call, we truncate the function call to the last #TO_USER
            contained_to_user = False
            if self.settings.to_user_sequence in func_seq_truncated and not func_seq_truncated.endswith(end_seq):
                func_seq_truncated = func_seq_truncated.rsplit(self.settings.to_user_sequence, 1)[0].strip()
                contained_to_user = True
            func_seq_truncated=func_seq_truncated.replace("$$$","")
            trunced_text = ret_text.replace(func_seq, "").replace(self.settings.function_sequence[0], "").replace(
                self.settings.function_sequence[1], "").strip()
            if not self.settings.to_user_sequence in func_seq_truncated and not contained_to_user and trunced_text != "":
                api.display_in_chat({"role": "assistant", "content": trunced_text, "code": func_seq_truncated})

            return_from_code = self.code_callback(func_seq_truncated)
            if isinstance(return_from_code, Exception):
                raise return_from_code
            if contained_to_user:
                func_seq_truncated += "\n"+self.settings.to_user_sequence
            kwarg_dic = {"code": func_seq_truncated, "return_value": return_from_code}#str(return_from_code)[-1500:] if return_from_code else "None (Function executed without return value and without error)"}

            if trunced_text != "":
                self.conversation_history.add_entry(change_latex_delimiters(trunced_text),
                                                    ConversationRoles.ASSISTANT, **kwarg_dic)
            else:
                self.conversation_history.add_entry("", ConversationRoles.ASSISTANT, **kwarg_dic)
            if self.settings.to_user_sequence in func_seq or contained_to_user or not return_from_code:
                if not return_from_code:
                    return self.conversation_history, metadata
                else:
                    self.conversation_history.tracker[-1]["content"] = str(return_from_code)
                    return self.conversation_history, metadata
            else:
                # if trunced_text != "":
                #     api.display_in_chat({"role":"assistant","content":trunced_text,**kwarg_dic})#.display_in_chat(text=trunced_text, role="partial")
                with open(rixaplugin.settings.WORKING_DIRECTORY + "/chat_tmp.txt", "a") as f:
                    text = "\n\n\n##########\nCODE WAS CALLED WITHOUT ERROR:\n"
                    text += f"CODE: {func_seq}\nRETURN VALUE: {repr(return_from_code)}\n##########"
                    f.write(text)
                self.create_completion_plugin(None, context=context, func_list=func_list,
                                              code_calls=code_calls, username=username, metadata=metadata)


        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            alm_logger.exception(f"Exception occurred in code:\n{func_seq}\n{tb}")

            # api.display(html="<h2>An exception occurred during an attempted code call</h2><code>" + tb.replace("\n", "<br>")[-2000:] + "</code>")
            api.show_message("An error occured. RIXA will try to fix it.")
            if self.settings.to_user_sequence in func_seq:
                kwarg_dic = {"code": func_seq, "return_value": "EXECUTION FAILED. REASON: " + str(e)}
                trunced_text = ret_text.replace(func_seq, "").replace(self.settings.function_sequence[0], "").replace(
                    self.settings.function_sequence[1], "").strip()
                if trunced_text != "":
                    self.conversation_history.add_entry(change_latex_delimiters(trunced_text),
                                                        ConversationRoles.ASSISTANT, **kwarg_dic)
                    #api.display_in_chat(text=change_latex_delimiters(trunced_text), role="partial")
                    api.display_in_chat({"role":"assistant","content":trunced_text})
                else:
                    # api.display_in_chat(text="An error has occurred. I will try to fix it", role="partial")
                    api.display_in_chat({"role":"assistant","content":"An error has occurred. I will try to fix it"})
                    self.conversation_history.add_entry("", ConversationRoles.ASSISTANT, **kwarg_dic)
            else:
                # api.display_in_chat(text="An error has occurred. I will try to fix it", role="partial")
                api.display_in_chat({"role": "assistant", "content": "An error has occurred. I will try to fix it"})
                self.conversation_history.add_entry(role=ConversationRoles.ASSISTANT, code=func_seq.strip(),
                                                    return_value="EXECUTION FAILED. REASON:\n" + repr(e))
            with open(rixaplugin.settings.WORKING_DIRECTORY + "/chat_tmp.txt", "a") as f:
                text = "\n\n\n*************\nCODE WAS CALLED WITH ERROR:\n"
                text += f"CODE: {func_seq}\nERROR:\n{tb}"
                f.write(text)
            self.create_completion_plugin(None, context=context, func_list=func_list, code_calls=code_calls, metadata=metadata)
        end = timer()
        self.finish_meta["total_finish_time"] = end - start
        return self.conversation_history, metadata

    # this could be improved by using raw token numbers. Then comparison to token number would be possible. would remedy whitespace issue
    # but that would break closed source compatability
    def create_generator(self, text_obj=None, verbose=None, enable_function_calls=None, chat=True,
                         token_prob_delta=None,
                         token_prob_abs=None, handle_functions=True, stop=None, **kwargs):
        """
        Streaming version of create_generator. Returns generator with automatic prompt building, function calling etc.

        :param text: Prompt or prompt obj
        :param max_tokens: maximum tokens generated in completion
        :param stop: Additional stop sequences
        :param chat: Only applicable for pure text models. Will pass just text to model without building a history.
        :param keep_dict: If library or API returns something else than raw tokens, whether to return native format
        :param token_prob_delta: dict, relative added number for token logits
        :param token_prob_abs: dict, Absolute logits for tokens
        :param log_probs: int, when not None return the top X log probs and their tokens
        :param handle_functions: Whether to call functions or return list with attemptedcalls
        :param kwargs: kwargs
        :return: completion
        """
        self.finish_meta = dict(self._finish_meta_template)
        self.raw_generated_text = ""
        if not self.settings.prompt_obj_is_str:
            chat = True
        if enable_function_calls is None:
            enable_function_calls = self.settings.global_enable_function_calls

        start = timer()
        if not verbose:
            verbose = self.verbose
        if not stop:
            stop = []

        if isinstance(stop, str):
            stop = [stop]

        if enable_function_calls:
            if not chat:
                warn("Enabling function calls in non-chat scenario can lead to strange results.")
            stop.append(self.settings.function_sequence[1])

        add_kwargs = {"stop": stop, "token_prob_delta": token_prob_delta, "token_prob_abs": token_prob_abs}

        if chat:
            if text_obj:
                self.add_tracker_entry(text_obj, ConversationRoles.USER)
            prompt_obj = self.build_prompt()
            self.prompt = prompt_obj
            if self.settings.prompt_obj_is_str and self.settings.include_conv_id_as_stop:
                stop.append(self.symbols["USER"])
                stop.append(self.symbols["ASSISTANT"])
            try:
                token_generator = self.create_native_generator(prompt_obj, **add_kwargs,
                                                               **kwargs)  # return_factory(prompt_obj)
            except Exception as e:
                self.pop_entry()
                raise e

        elif text_obj and self.settings.prompt_obj_is_str:
            try:
                token_generator = self.create_native_generator(text_obj, **add_kwargs, **kwargs)
            except Exception as e:
                self.pop_entry()
                raise e

        if not chat and not text_obj:
            raise Exception("No prompt given!")

        sequences = []
        for i, x in self.settings.preserved_sequences.items():
            x["type"] = i
            sequences.append(x)
        if chat and enable_function_calls:
            sequences.append({"type": "function", "start": self.settings.function_sequence[0],
                              "end": self.settings.function_sequence[1]})

        buffer = []
        buffer_logits = []
        buffer_str = ""
        sequence_tokens = []
        start_sequence = None
        end_sequence = None
        in_sequence = False
        yield_type = "token"

        generator_list = [token_generator]

        cleanup_sentinel = object()
        last_generated_text = ""

        def token_generator_with_insertions():
            while True:
                if len(generator_list) != 0:
                    cur_gen = generator_list.pop()
                    yield from cur_gen
                    yield cleanup_sentinel, None
                else:
                    break

        caught_err = io.StringIO()
        with contextlib.redirect_stderr(caught_err):

            for token, logits_or_none in token_generator_with_insertions():
                if not token is cleanup_sentinel:
                    self.raw_generated_text += token
                    last_generated_text += token
                    buffer.append(token)
                    buffer_logits.append(logits_or_none)
                    buffer_str = ''.join(buffer)
                if not in_sequence:
                    for sequence in sequences:
                        if buffer_str.strip().startswith(sequence['start']):
                            in_sequence = True
                            start_sequence = sequence['start']
                            end_sequence = sequence['end']
                            sequence_tokens.extend(buffer)
                            yield_type = start_sequence + "XYZ" + end_sequence if not "type" in sequence else sequence[
                                "type"]
                            buffer = []
                            break
                    if not in_sequence and len(buffer) > len(max(sequences, key=lambda s: len(s['start']))['start']):
                        yield buffer.pop(0), yield_type, buffer_logits.pop(0)
                else:
                    sequence_tokens.append(token)
                    if buffer_str.endswith(end_sequence):
                        in_sequence = False
                        if yield_type == "function":
                            raise NotImplementedError()
                        else:
                            yield ''.join(sequence_tokens), yield_type, logits_or_none
                        yield_type = "token"
                        sequence_tokens = []
                        buffer = []
                        buffer_logits = []
                if token is cleanup_sentinel:
                    if len(sequence_tokens) != 0:
                        sequence_tokens.pop()
                    buf_temp = buffer
                    buf_logits_temp = buffer_logits
                    buffer = []
                    buffer_logits = []
                    in_sequence = False

                    if not enable_function_calls:
                        if len(buf_temp) != 0:
                            sequence_tokens = []
                            yield "".join(buf_temp), None, buf_logits_temp
                    else:
                        status, seq, new_text = self._extract_and_handle_functions(self.raw_generated_text,
                                                                                   call_functions=handle_functions)
                        if status == ParseStatus.NO_FUNC_SEQUENCE_FOUND:
                            yield "".join(buf_temp), None, buf_logits_temp
                            # yield ''.join(sequence_tokens), "end", logits_or_none
                            break
                        if status != ParseStatus.PARSED_EXECUTED_OK:
                            self.finish_meta["finish_reason"] = "unparseable function call"
                            yield ''.join(sequence_tokens), "failed function call", logits_or_none
                            break
                        sequence_tokens = []
                        try:
                            stop.remove(self.preserved_sequences["functions"]["end"])
                        except:
                            pass
                        sequences = [d for d in sequences if d.get('type') != 'function']
                        prompt_obj = self.build_prompt()
                        token_generator_funcs = self.create_native_generator(prompt_obj, **add_kwargs, **kwargs)
                        generator_list.append(token_generator_funcs)
                        enable_function_calls = False
                        last_generated_text = ""
        if chat:
            self.add_tracker_entry(last_generated_text, ConversationRoles.ASSISTANT)
        end = timer()
        self.finish_meta["total_finish_time"] = end - start

    def build_prompt_as_str(self, new_lines_per_role=1, new_lines_afer_role=0, block_gen_prefix=False, raw=False,
                            include_system_msg=True, use_build_prompt=False, max_index=0):
        """
        Build a prompt in string form
        :param new_lines_per_role: Newlines after Role+message
        :param new_lines_afer_role: Newlines after role
        :param block_gen_prefix: Whether to add generation prefix. Normally this leads a prompt to end with e.g.
        Assistant: so that the model does not continue to write as a user.
        :param raw: If true don't resolve symbols
        :return: str
        """
        if use_build_prompt:
            prompt_obj = self.build_prompt()
            prompt_str = ""
            for i in prompt_obj[:max_index] if max_index != 0 else prompt_obj:
                if not include_system_msg:
                    if i["role"] == "system":
                        continue
                prompt_str += f"{i['role']}:{i['content']}\n"
            return prompt_str

        def rep_sym(str, entry=None):
            return str if raw else self.replace_symbols(str, entry)

        prompt = ""
        after_role = "\n" * new_lines_afer_role if new_lines_afer_role != 0 else " "
        if include_system_msg:
            if self.conversation_history.system_message and self.conversation_history.system_message != "":
                prompt += f"{self.symbols['SYSTEM']}:{after_role}{rep_sym(self.system_msg)}" + "\n" * new_lines_per_role

        for i in self.conversation_history.tracker[:max_index] if max_index != 0 else self.conversation_history.tracker:
            role = self.symbols[str(i["role"])]
            if "code" in i and i["code"]:
                code_str = "CODE_START\n" + i["code"] + "\nCODE_START"
                if "return_value" in i and i["return_value"]:
                    code_str += "\nRETURN:\n" + i["return_value"]
                else:
                    code_str += "\nRETURN:\nNone"
                prompt += f"{role}:{after_role}{code_str}" + "\n" * new_lines_per_role
            else:
                prompt += f"{role}:{after_role}{rep_sym(i['content'], i)}" + "\n" * new_lines_per_role
        if block_gen_prefix:
            prompt = prompt[:-1]
        if not block_gen_prefix and self.settings.generation_prefix and self.settings.generation_prefix != "":
            prompt += rep_sym(self.settings.generation_prefix)
        return prompt

    # for i in conv_history:
    #     if "code" in i and i["code"]:
    #         code_str = "CODE_START\n" + i["code"] + "\nCODE_START"
    #         if "return_value" in i and i["return_value"]:
    #             code_str += "\nRETURN:\n" + i["code_return_val"]
    #         else:
    #             code_str += "\nRETURN:\nNone"
    #         prompt.append({"role": self.symbols["SYSTEM"], "content": code_str})
    #     else:
    #         prompt.append({"role": self.symbols[str(i["role"])], "content": self.replace_symbols(i["content"], i)})

    def _text_callback(self, text):
        pass
        # msg = text.value
        # self.text_input.value="Loading...."
        # self.text_input.disabled=True
        # self.get_response()
        # self.update()
        # self._display_chat()

    def update_gui(self):
        if not self.jupyter_gui:
            # warn("GUI not initialized. This has no effect")
            self.init_gui()
            return
        if "markdown" not in globals():
            from markdown import markdown
            from IPython.display import HTML, clear_output, display
            globals()["clear_output"] = clear_output
            globals()["HTML"] = HTML
            globals()["display"] = display
            globals()["markdown"] = markdown
        gl = globals()
        messages = ""
        for i in self.conversation_history.tracker:
            text = i["content"]
            text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
            text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text)
            text = gl["markdown"](text)
            tok = f""
            if "tokens" in i:
                tok = f"{i['tokens']}t"
            messages += f"""
            <div class="{'message received' if i['role'] == ConversationRoles.ASSISTANT else ('message sent' if i['role'] == ConversationRoles.USER else 'message system')}">
                <div class="metadata">
                    <span class="sender">{self.symbols[str(i["role"])]}</span>
                    <span class="time">{tok}</span>
                </div>
                <div class="text">
                    {text}
                </div>
            </div>"""

        chat_protocoll = _chat_html.replace("INSERTHEREXYZ", messages)
        with self._gui_chat_history:
            gl["clear_output"]()
            gl["display"](gl["HTML"](chat_protocoll))
        # with self.token_count:
        #     clear_output()
        #     display(HTML(f"{self.data['total_token']}t, {round_two_nonzero(self.data['total_dollar'])}$"))
        # self.text_input.value=""
        # self.text_input.disabled=False

    def _gui_on_switch(self, btn):
        self.conversation_history.invert_roles()
        self.update_gui()

    def _gui_clear_tracker(self, btn):
        self.conversation_history.tracker = []
        self.update_gui()

    def _gui_del_message(self, btn):
        self.pop_entry()
        self.update_gui()

    def autoconverse(self, interactions=3):
        for i in range(interactions):
            print(f"\n\nInteraction {i}\n-----------\n")
            gen = self.create_generator()
            for i in gen:
                print(i[0], end="")
            self.conversation_history.invert_roles()

    def init_gui(self, num_messages_included=4, monkeypatch=True):
        if self.jupyter_gui:
            warn("GUI can't be re-inited")
            return
        try:
            from ipywidgets import GridspecLayout, VBox, Box
            from ipywidgets import Output
            import ipywidgets as widgets
            from IPython.display import HTML, clear_output, display
            from markdown import markdown
            globals()["clear_output"] = clear_output
            globals()["HTML"] = HTML
            globals()["display"] = display
            globals()["markdown"] = markdown
        except:
            raise Exception("Visualization requires ipywidgets and markdown, which is not installed!")
        # grid = GridspecLayout(2, 3)

        clear_tracker_button = widgets.Button(
            description='Clear history',
            disabled=False,
            button_style='',
        )
        delete_button = widgets.Button(
            description='Delete last',
            disabled=False,
            button_style='',
        )
        switch_button = widgets.Button(
            description='Switch roles',
            disabled=False,
            button_style='',
        )
        switch_button.on_click(self._gui_on_switch)
        delete_button.on_click(self._gui_del_message)
        clear_tracker_button.on_click(self._gui_clear_tracker)

        buttons = Box([clear_tracker_button, delete_button, switch_button])

        self._gui_chat_history = Output()

        grid = VBox([self._gui_chat_history, buttons])

        self.jupyter_gui = True
        self.update_gui()
        original_completion = self.add_tracker_entry

        def tracker_patch(*args, **kwargs):
            original_completion(*args, **kwargs)
            self.update_gui()

        setattr(self, "add_tracker_entry", tracker_patch)

        display(grid)


_chat_html = """
<html>
<head>
	<title>Chat Example</title>
	<style>
		.container {
			display: flex;
			flex-direction: column;
			height: 100%;
			padding: 10px;
			background-color: #f2f2f2;
			font-size: 16px;
			font-family: Arial, sans-serif;
            overflow: scroll;
		}
		.message {
			display: flex;
			flex-direction: column;
			align-items: flex-start;
			max-width: 70%;
			margin-bottom: 10px;
			padding: 10px;
			border-radius: 10px;
			background-color: #fff;
			box-shadow: 0px 1px 3px rgba(0,0,0,0.2);
            white-space: pre-line;
		}
		.message.sent {
			align-self: flex-end;
			background-color: #dcf8c6;
		}
		.message.received {
			align-self: flex-start;
			background-color: #e6e6e6;
		}
        .message.system {
			align-self: center;
			background-color: orange;
		}
		.message .metadata {
			display: flex;
			flex-direction: row;
			align-items: center;
			justify-content: flex-start;
			color: #888;
			font-size: 12px;
			margin-bottom: 5px;
		}
		.message .metadata .time {
			margin-left: 5px;
		}
		.message .text {
			color: #333;
			font-size: 14px;
			line-height: 1.2;
		}
	</style>
</head>
<body>
<div style="height:45em">
	<div class="container">
        INSERTHEREXYZ
</div>
<div id="bottom"></div>
function bottom() {
    document.getElementById( 'bottom' ).scrollIntoView();
    window.setTimeout( function () { top(); }, 2000 );
};

bottom();
</body>
</html>

"""
