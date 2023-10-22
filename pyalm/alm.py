from abc import abstractmethod
import psutil
import ast
import enum
import re
from timeit import default_timer as timer
from pylot import python_parsing
from typing import Type
from functools import partial
from warnings import warn
import contextlib
import io
import yaml
import os


class ConversationRoles(enum.Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"

    def __str__(self) -> str:
        return self.value


def conversation_role_representer(dumper, data):
    return dumper.represent_scalar('!ConversationRole', str(data))


def conversation_role_constructor(loader, node):
    value = loader.construct_scalar(node)
    return _get_enum_value(value, ConversationRoles)


yaml.add_representer(ConversationRoles, conversation_role_representer)
yaml.add_constructor('!ConversationRole', conversation_role_constructor)


class FunctionFormat(enum.Enum):
    PYDOC = "PYDOC"
    JSON = "JSON"
    MODEL_SPECIFIC = "MODEL_SPECIFIC"

    def __str__(self) -> str:
        return self.value


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


# TODO move verbose into settings
class ALM:
    """
    Base class. Don't instantiate on its own
    """

    def __init__(self, model_path_or_name, n_ctx=2048, verbose=0):
        self.model = model_path_or_name
        self.verbose = verbose
        # self.atomic_sequences = {"functions": ["+++", "---"],  # ["➡️", "⬅️"],["❬", "❭"]
        #                          "latex_double": ["$$", "$$"],
        #                          "latex_single": ["$", "$"]}
        func_inclusion_message = """[[LIST_OF_FUNCTIONS]]
Above you is a list of functions you can call. To call them enclose them with [[FUNC_DELIMITER_START]] and end the call with [[FUNC_DELIMITER_END]].
The entire sequence must be correct! Do not e.g. leave out the [[FUNC_DELIMITER_END]].
This
[[FUNC_DELIMITER_START]]foo(bar=3)[[FUNC_DELIMITER_END]]
would call the function foo with bar=3. The function(s) will return immediately. The values will be in the inverse sequence of the function enclosement.  
You can only call the functions listed.
You can and HAVE TO call functions during the text response not in a a separate response!
Before you call a function please inform the user so he is aware of possible waiting times.
"""
        self.preserved_sequences = {"functions": {"start": "+++", "end": "---", "type": "function"},
                                    # "latex_double": {"start": "$$", "end": "$$", "name": "latex1"}
                                    }
        """Dictionary of sequences that only get yielded as whole
        """
        self.symbols = {"FUNC_DELIMITER_START": self.preserved_sequences["functions"]["start"],
                        "FUNC_DELIMITER_END": self.preserved_sequences["functions"]["end"],
                        "ASSISTANT": "Assistant", "USER": "User", "SYSTEM": "System",
                        "FUNC_INCLUSION_MESSAGE": func_inclusion_message, "LIST_OF_FUNCTIONS": "NO FUNCTIONS AVAILABLE"}
        """
        Variable symbols that will get replaced when building prompt. Either string or function pointer 
        """
        self.base_settings = {"GENERATION_PREFIX": "[[ASSISTANT]]: ", "FUNCTIONS_ENABLED": True,
                              "FUNCTION_AUTOINTEGRATION": True,
                              "function_integration_template": "\n[[FUNC_DELIMITER_START]][[FUNCTION_SEQUENCE]][[FUNC_DELIMITER_END]]\n[[FUNC_DELIMITER_END]][[FUNCTION_RETURN_VALUE]][[FUNC_DELIMITER_START]]"}
        """Default values for settings"""
        self.settings = dict(self.base_settings)

        self.func_map = {}
        """Function names and the callable functions available for the model"""

        self.available_functions = []
        self.conv_history = []

        self.system_msg = {}
        self.generated_text = ""


        self.prompt_text_is_str = False
        self.parse_status = ParseStatus.UNDEFINED

    def adopt_from_alm(self, other: Type['ALM']):
        """
        Copy state from other ALM into this. This is not a deep copy!

        :param other: Other ALM
        """
        self.conv_history = other.conv_history
        self.system_msg = other.system_msg
        self.verbose = other.verbose
        self.available_functions = other.available_functions
        self.settings.update(other.settings)

    def _repl(self, match, text=None, temp_symbols={}):
        """
        Callable for re.sub in _replace_symbols

        :param match: regex match
        :param text: whole text
        :param temp_symbols: additional symbols
        :return: replacement
        """
        if match[1] in self.symbols:
            val = self.symbols[match[1]]
            if isinstance(val, str):
                return self.symbols[match[1]]
            else:
                try:
                    return val(match, text, temp_symbols)
                except Exception as e:
                    raise Exception("An error occurred while trying to substitute symbols for prompt:\n" + str(e))
        if match[1] in temp_symbols:
            val = temp_symbols[match[1]]
            if isinstance(val, str):
                return temp_symbols[match[1]]
            else:
                try:
                    return val(match, text, temp_symbols)
                except Exception as e:
                    raise Exception("An error occurred while trying to substitute symbols for prompt:\n" + str(e))
        return f"#KEY_MISSING: {match[1]}#"

    def _build_func_template(self, match, text, temp_symbols):
        return self._replace_symbols(self.settings["function_integration_template"], temp_symbols=temp_symbols)

    def _replace_symbols(self, text, entry=None, temp_symbols=None):
        """
        Replace symbols in an conv history entry or text

        :param text: text with symbols in it
        :param entry: optional, conv history entry to use for replacement
        :param temp_symbols: additional symbols to be replaced
        :return: text with substitutions
        """
        if not temp_symbols:
            temp_symbols = {}
            if entry:
                if "function_calls" in entry:
                    if "original_call" in entry["function_calls"]:
                        temp_symbols["FUNCTION_SEQUENCE"] = entry["function_calls"]["original_call"]
                    if "return" in entry["function_calls"]:
                        temp_symbols["FUNCTION_RETURN_VALUE"] = entry["function_calls"]["return"]
        if self.settings["FUNCTION_AUTOINTEGRATION"]:
            temp_symbols["FUNCTION_CALL"] = self._build_func_template
        pattern = r'\[\[(.*?)\]\]'
        text = re.sub(pattern, lambda match: self._repl(match, text, temp_symbols), text)
        return text

    def save_state(self, path=None):
        """
        Saves the ALMs entire state (excluding the model itself)

        :param path: Where to save. If not specified string is returned
        :return: None or state as yaml
        """
        state = {"system_msg": self.system_msg, "conv_history": self.conv_history, "settings": self.settings,
                 "symbols": self.symbols, "preserved_sequences": self.preserved_sequences}
        yaml_str = yaml.dump(state, sort_keys=False)
        if not path:
            return yaml_str

        with open(path, "w") as f:
            f.write(yaml_str)

    # TODO implement string return
    def save_history(self, path=None):
        """
        Saves the ALMs conversation history

        :param path: Where to save. If not specified string is returned
        :return: None or history as yaml like string
        """
        yaml_str = yaml.dump(self.conv_history)
        if not path:
            return yaml_str

        with open(path, "w") as f:
            f.write(yaml_str)

    def load_history(self, path_or_text):
        """
        Loads a conversation history

        :param path_or_text: Either path to a file or a yaml like string
        """
        if os.path.exists(path_or_text):
            with open(path_or_text, "r") as f:
                str = f.read()
        else:
            str = path_or_text

        self.conv_history = yaml.full_load(str)

    def set_system_message(self, msg, prepend_function_support=True):
        """
        Change the system message

        :param msg: new message
        :param prepend_function_support: Automatically change message to include function calling
        """
        if prepend_function_support:
            msg = self.symbols["FUNC_INCLUSION_MESSAGE"] + msg
        self.system_msg["content"] = msg

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

    @abstractmethod
    def build_prompt(self, preserve_flow=False):
        """
        Build prompt in format native to library

        :param preserve_flow: Block suffix for purely text based models
        :return: prompt obj
        """
        raise NotImplementedError()

    def reset_tracker(self):
        """
        Remove all tracker entries
        """
        self.conv_history = []

    def add_tracker_entry(self, role, content=None, meta=None, function_calls=None, context=None, feedback=None,
                          sentiment=None, add_keys=None):
        """
        Add a new entry to the tracker. More info is in
        https://github.com/finnschwall/PyALM/blob/main/format_specifications.md
        """

        role = _get_enum_value(role, ConversationRoles)

        msg = {"role": role}
        excl = ["msg", "role", "add_keys", "excl", "loc", "self"]
        loc = locals()
        for i in loc:
            if i in excl:
                continue
            if loc[i]:
                msg[i] = loc[i]
        if add_keys:
            msg = msg | add_keys
            msg = {"role": role}
        self.conv_history.append(msg)

    # @staticmethod
    # def functions_to_dict(functions):

    def register_functions(self, functions):
        """
        Add functions to be callable by the model

        :param functions: Function or list of functions
        :return:
        """
        if not isinstance(functions, list):
            functions = [functions]
        if len(functions) == 0:
            raise Exception("List is empty")
        dic_list = []
        self.symbols["LIST_OF_FUNCTIONS"] = ""
        try:
            for i in functions:
                func_as_dic = python_parsing.function_signature_to_dict(i)
                pydoc = python_parsing.generate_python_doc(func_as_dic["name"], func_as_dic)
                self.symbols["LIST_OF_FUNCTIONS"] += pydoc + "\n"
                func_as_dic["pydoc"] = pydoc
                func_as_dic["callback"] = i
                dic_list.append(func_as_dic)
                self.func_map[func_as_dic["name"]] = i
            self.available_functions = dic_list
        except Exception as e:
            self.symbols["LIST_OF_FUNCTIONS"] = "No functions available"
            raise e

    def create_completion(self, text_obj=None, verbose=None, enable_function_calls=None, chat=True,
                          token_prob_delta=None, token_prob_abs=None, handle_functions=True, stop=[], **kwargs):
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
        self.finish_meta["function_call"] = {"found": False}
        self.generated_text = ""
        if not self.prompt_text_is_str:
            chat = True
        if enable_function_calls is None:
            enable_function_calls = self.settings["FUNCTIONS_ENABLED"]

        start = timer()
        if not verbose:
            verbose = self.verbose

        if isinstance(stop, str):
            stop = [stop]

        if enable_function_calls:
            if not chat:
                warn("Enabling function calls in non-chat scenario can lead to strange results.")
            stop.append(self.preserved_sequences["functions"]["end"])

        add_kwargs = {"stop": stop, "token_prob_delta": token_prob_delta, "token_prob_abs": token_prob_abs}

        if chat:
            if text_obj:
                self.add_tracker_entry(ConversationRoles.USER, content=text_obj)
            prompt_obj = self.build_prompt()
            self.prompt = prompt_obj
            if self.prompt_text_is_str:
                stop.append(self.symbols["USER"])
                stop.append(self.symbols["ASSISTANT"])
            ret_text = self.create_native_completion(prompt_obj, **add_kwargs, **kwargs)  # return_factory(prompt_obj)

        elif text_obj and self.prompt_text_is_str:
            ret_text = self.create_native_completion(text_obj, **add_kwargs, **kwargs)

        if not chat and not text_obj:
            raise Exception("No prompt given!")

        self.generated_text += ret_text
        if not enable_function_calls:
            if chat:
                self.add_tracker_entry(ConversationRoles.ASSISTANT, content=ret_text)
            end = timer()
            self.finish_meta["total_finish_time"] = end - start
            return self.generated_text

        status, seq, new_text = self._extract_and_handle_functions(ret_text, call_functions=handle_functions)

        if status != ParseStatus.NO_FUNC_SEQUENCE_FOUND:
            self.finish_meta["function_call"]["found"] = True
            self.finish_meta["function_call"]["sequence"] = seq

        if status != ParseStatus.PARSED_EXECUTED_OK:
            if chat:
                self.add_tracker_entry(ConversationRoles.ASSISTANT, content=ret_text)
            end = timer()
            self.finish_meta["total_finish_time"] = end - start
            return self.generated_text
        self.generated_text = new_text + "\n"
        try:
            stop.remove(self.preserved_sequences["functions"]["end"])
        except:
            pass
        prompt_obj = self.build_prompt()
        ret_text = self.create_native_completion(prompt_obj, **add_kwargs, **kwargs)

        self.generated_text += ret_text

        if chat:
            self.add_tracker_entry(ConversationRoles.ASSISTANT, content=ret_text)
        end = timer()
        self.finish_meta["total_finish_time"] = end - start
        return self.generated_text

    def _extract_and_handle_functions(self, text, call_functions=True):
        """
        Extract function sequences from text and parse and potentially execute them

        :param text:
        :param call_functions: Call or just collect calls and return
        :return:
        """
        start_seq = self.preserved_sequences["functions"]["start"]
        end_seq = self.preserved_sequences["functions"]["end"]
        pattern = re.escape(start_seq) + '(.*?)' + re.escape(end_seq)
        matches = [(m.group(1), m.span()) for m in re.finditer(pattern, text, re.DOTALL)]

        if len(matches) == 0:
            text += end_seq
            pattern = re.escape(start_seq) + '(.*?)' + re.escape(end_seq)
            matches = [(m.group(1), m.span(), m) for m in re.finditer(pattern, text, re.DOTALL)]

            if len(matches) == 0:
                self.parse_status = ParseStatus.NO_FUNC_SEQUENCE_FOUND
                return ParseStatus.NO_FUNC_SEQUENCE_FOUND, None, None
        func_seq = matches[0][0].strip()
        try:
            ast_obj = ast.parse(func_seq)
        except Exception as e:
            if call_functions:
                raise Exception("Models doing stuff wrong is not yet correctly handled")
            self.parse_status = ParseStatus.UNPARSEABLE_FUNC_FOUND
            return ParseStatus.UNPARSEABLE_FUNC_FOUND, func_seq, e

        if not call_functions:
            visitor = python_parsing.CodeVisitor(collect=True)
            visitor.visit(ast_obj)
            self.parse_status = ParseStatus.PARSED_DICT_RETURN
            return ParseStatus.PARSED_DICT_RETURN, func_seq, visitor.collection

        visitor = python_parsing.CodeVisitor(self.func_map)
        visitor.visit(ast_obj)
        if "__call_res__" in visitor.variables:
            ret_val = visitor.variables["__call_res__"]
        else:
            ret_val = "NO RETURN VALUE"

        loc = text.find(func_seq)
        loc -= len(end_seq)
        # new_assistant_text = text[:loc] + "[[ORIGINAL_CALL]]"+
        self.add_tracker_entry(ConversationRoles.ASSISTANT, content=text[:loc] + "[[FUNCTION_CALL]]",
                               function_calls={"original_call": func_seq, "return": ret_val})
        self.parse_status = ParseStatus.PARSED_EXECUTED_OK

        return ParseStatus.PARSED_EXECUTED_OK, func_seq, text[:loc]

    # this could be improved by using raw token numbers. Then comparison to token number would be possible. would remedy whitespace issue
    # but that would break closed source compatability
    def create_generator(self, text_obj=None, verbose=None, enable_function_calls=None, chat=True,
                         token_prob_delta=None,
                         token_prob_abs=None, handle_functions=True, stop=[], **kwargs):
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
        self.finish_meta["function_call"] = {"found": False}
        self.generated_text = ""
        if not self.prompt_text_is_str:
            chat = True
        if enable_function_calls is None:
            enable_function_calls = self.settings["FUNCTIONS_ENABLED"]

        start = timer()
        if not verbose:
            verbose = self.verbose

        if isinstance(stop, str):
            stop = [stop]

        if enable_function_calls:
            if not chat:
                warn("Enabling function calls in non-chat scenario can lead to strange results.")
            stop.append(self.preserved_sequences["functions"]["end"])

        add_kwargs = {"stop": stop, "token_prob_delta": token_prob_delta, "token_prob_abs": token_prob_abs}

        if chat:
            if text_obj:
                self.add_tracker_entry(ConversationRoles.USER, content=text_obj)
            prompt_obj = self.build_prompt()
            self.prompt = prompt_obj
            if self.prompt_text_is_str:
                stop.append(self.symbols["USER"])
                stop.append(self.symbols["ASSISTANT"])
            token_generator = self.create_native_generator(prompt_obj, **add_kwargs,
                                                           **kwargs)  # return_factory(prompt_obj)

        elif text_obj and self.prompt_text_is_str:
            token_generator = self.create_native_generator(text_obj, **add_kwargs, **kwargs)

        if not chat and not text_obj:
            raise Exception("No prompt given!")

        sequences = []
        for i, x in self.preserved_sequences.items():
            if not chat and i == "functions":
                continue
            x["type"] = i
            sequences.append(x)

        buffer = []
        buffer_logits = []
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
        tok_list = []
        with contextlib.redirect_stderr(caught_err):
            for token, logits_or_none in token_generator_with_insertions():
                if not token is cleanup_sentinel:
                    # print(token,end="")
                    self.generated_text += token
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
                        if len(buffer) != 0:
                            sequence_tokens = []
                            yield "".join(buf_temp), None, buf_logits_temp
                    else:
                        status, seq, new_text = self._extract_and_handle_functions(self.generated_text,
                                                                                   call_functions=handle_functions)
                        # print()
                        # print(sequence_tokens)
                        # print(status,seq,new_text)
                        if status == ParseStatus.NO_FUNC_SEQUENCE_FOUND:
                            yield ''.join(sequence_tokens), "end", logits_or_none
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
            self.add_tracker_entry(ConversationRoles.ASSISTANT, content=last_generated_text)
        end = timer()
        self.finish_meta["total_finish_time"] = end - start
        # print(tok_list)

    def build_prompt_as_str(self, new_lines_per_role=1, new_lines_afer_role=0, block_gen_prefix=False, raw = False):
        """
        Build a prompt in string form
        :param new_lines_per_role: Newlines after Role+message
        :param new_lines_afer_role: Newlines after role
        :param block_gen_prefix: Whether to add generation prefix
        :param raw: If true don't resolve symbols
        :return:
        """
        def rep_sym(str,entry=None):
            return str if raw else self._replace_symbols(str,entry)

        prompt = ""
        after_role = "\n" * new_lines_afer_role if new_lines_afer_role != 0 else " "
        if "content" in self.system_msg:
            prompt += f"{self.symbols['SYSTEM']}:{after_role}{rep_sym(self.system_msg['content'])}" + "\n" * new_lines_per_role

        for i in self.conv_history:
            role = self.symbols[str(i["role"])]
            prompt += f"{role}:{after_role}{rep_sym(i['content'], i)}" + "\n" * new_lines_per_role
        if block_gen_prefix:
            prompt = prompt[:-1]
        if not block_gen_prefix and "GENERATION_PREFIX" in self.settings and self.settings["GENERATION_PREFIX"] != "":
            prompt += rep_sym(self.settings["GENERATION_PREFIX"])
        return prompt


def _get_enum_value(input_value, enum_type):
    if isinstance(input_value, str):
        try:
            return enum_type[input_value.upper()]
        except KeyError:
            raise ValueError(f"'{input_value}' not found in {enum_type.__name__} enum.")
    elif isinstance(input_value, enum_type):
        return input_value
    else:
        raise TypeError(f"Invalid input type. Expected enum value or string, got {type(input_value).__name__}.")
