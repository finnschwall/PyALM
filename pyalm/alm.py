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
import dataclasses as dc  # import dataclass, asdict, field
from abc import ABC, abstractmethod


@dc.dataclass
class DataYAML(ABC):

    def to_dict(self):
        return dc.asdict(self)

    @classmethod
    def from_dict(cls, dict_obj):
        return cls(**dict_obj)

    def save_to_yaml(self, path=None):
        yaml_str = yaml.dump(self.to_dict(), sort_keys=False)
        if not path:
            return yaml_str
        with open(path, "w") as f:
            f.write(yaml_str)

    @classmethod
    def load_from_yaml(cls, path_or_text):
        if os.path.exists(path_or_text):
            with open(path_or_text, "r") as f:
                data = f.read()
        else:
            data = path_or_text
        data = yaml.full_load(data)
        instance = cls.from_dict(data)
        return instance


@dc.dataclass(kw_only=True)
class ConversationTracker(DataYAML):
    system_message: str = None
    user_info: dict = dc.field(default_factory=dict)
    tracker: list = dc.field(default_factory=list)

    def reset_tracker(self):
        temp = self.tracker
        self.tracker = []
        return temp

    def add_entry(self, role, content=None, meta=None, function_calls=None, feedback=None,
                  sentiment=None, add_keys=None):
        role = _get_enum_value(role, ConversationRoles)

        entry = {"role": role}
        if content:
            entry["content"] = content
        if meta:
            entry["meta"] = meta
        if function_calls:
            entry["function_calls"] = function_calls
        if feedback:
            entry["feedback"] = feedback
        if add_keys:
            entry = entry | add_keys
        self.tracker.append(entry)
        return entry


@dc.dataclass(kw_only=True)
class ALMSettings(DataYAML):
    verbose: int = 0
    preserved_sequences: dict = dc.field(
        default_factory=lambda: {"latex_double": {"start": "$$", "end": "$$", "name": "latex_double_dollar"}})
    function_sequence: tuple = dc.field(default_factory=lambda: ("+++", "---"))
    global_enable_function_calls: bool = True
    automatic_function_integration: bool = True
    function_integration_template: str = "\n[[FUNCTION_START]][[FUNCTION_SEQUENCE]][[FUNCTION_END]]\n" \
                                         "[[FUNCTION_END]][[FUNCTION_RETURN_VALUE]][[FUNCTION_START]]"
    generation_prefix: str = "[[ASSISTANT]]:"

    function_inclusion_instruction_system_msg = """[[LIST_OF_FUNCTIONS]]
Above you is a list of functions you can call. To call them enclose them with [[FUNCTION_START]] and end the call with [[FUNCTION_END]].
The entire sequence must be correct! Do not e.g. leave out the [[FUNCTION_END]].
This
[[FUNCTION_START]]foo(bar=3)[[FUNCTION_END]]
would call the function foo with bar=3. The function(s) will return immediately. The values will be in the inverse sequence of the function enclosement.  
You can only call the functions listed.
You can and HAVE TO call functions during the text response not in a a separate response!
Before you call a function please inform the user so he is aware of possible waiting times.
"""
    prompt_obj_is_str: bool = True


def _data_yaml_representer(dumper, data):
    return dumper.represent_dict({'class': type(data).__name__, 'data': data.to_dict()})


def _data_yaml_constructor(loader, node):
    data = loader.construct_dict(node)
    cls = globals()[data['class']]
    return cls.from_dict(data['data'])


for i in [DataYAML, ConversationTracker]:
    yaml.add_representer(i, _data_yaml_representer)
    yaml.add_constructor('!' + i.__name__, _data_yaml_constructor)


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


class Symbols(dict):
    def __setitem__(self, key, value):
        key = key.upper()
        if key == "FUNCTION_START" or key == "FUNCTION_END":
            raise KeyError(f"'{key}' is a reference. Change 'function_sequence' tuple in settings")
        if key == "FUNCTION_CALL":
            raise KeyError(f"'{key}' is a reference. Change 'function_integration_template' in settings")

        super().__setitem__(key, value)


# TODO move verbose into settings
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
        symbols.update(self._temp_symbols)
        return symbols

    @symbols.setter
    def symbols(self, system_msg):
        raise Exception("Add or modify symbols via 'user_symbols'")

    def __init__(self, model_path_or_name, verbose=0):
        self.model = model_path_or_name

        self.settings = ALMSettings(verbose=verbose)
        self.settings.verbose = verbose
        self.model_meta = {"model_name": model_path_or_name}

        self.conversation_history = ConversationTracker()

        self._built_in_symbols = {
            "FUNCTION_START": lambda match, symbols, text=None: self.settings.function_sequence[0],
            "FUNCTION_END": lambda match, symbols, text=None: self.settings.function_sequence[1],
            "ASSISTANT": "Assistant", "USER": "User", "SYSTEM": "System",
            "FUNCTION_CALL": lambda match, symbols, text=None: self.replace_symbols(
                self.settings.function_integration_template, temp_symbols=temp_symbols)}

        self.user_symbols = Symbols()
        """
        Variable symbols that will get replaced when building prompt. Either string or function pointer 
        """

        self._temp_symbols = {}
        # TODO modify PyLoT so that this works with doc, signature etc. included
        self.available_functions = {}
        """Function names and the callable functions available for the model"""

        self.raw_generated_text = ""

        self._finish_meta_template = {"function_call": {"found": False, "parse_status": ParseStatus.UNDEFINED},
                                      "finish_reason": "Unknown", "timings": {}, }
        self.finish_meta = dict(self._finish_meta_template)

    # TODO implement
    def pop_entry(self):
        """
        Remove last element from conversation tracker. Automatically takes care of e.g. split messages due to function calls.
        :return: The popped entry
        """
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
    def load_history(self, path_or_text):
        """
        Loads a conversation history

        :param path_or_text: Either path to a file or a yaml like string
        """
        self.conversation_history = ConversationTracker.load_from_yaml(path_or_text)

    def set_system_message(self, msg, prepend_function_support=True):
        """
        Change the system message

        :param msg: new message
        :param prepend_function_support: Automatically change message to include function calling
        """
        if prepend_function_support:
            msg = self.settings.function_inclusion_instruction_system_msg
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
        self.conversation_history.reset_tracker()

    def add_tracker_entry(self, role, content=None, meta=None, function_calls=None, feedback=None,
                          sentiment=None, add_keys=None):
        """
        Add a new entry to the tracker. More info is in
        https://github.com/finnschwall/PyALM/blob/main/format_specifications.md
        """
        loc_dic = locals()
        del loc_dic["self"]
        del loc_dic["role"]
        self.conversation_history.add_entry(role, **loc_dic)

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
        self._temp_symbols["LIST_OF_FUNCTIONS"] = ""
        try:
            for i in functions:
                func_as_dic = python_parsing.function_signature_to_dict(i)
                pydoc = python_parsing.generate_python_doc(func_as_dic["name"], func_as_dic)
                self._temp_symbols["LIST_OF_FUNCTIONS"] += pydoc + "\n"
                func_as_dic["pydoc"] = pydoc
                func_as_dic["callback"] = i
                dic_list.append(func_as_dic)
                self.available_functions[func_as_dic["name"]] = i
        except Exception as e:
            self.symbols["LIST_OF_FUNCTIONS"] = "No functions available"
            raise e

    def create_completion(self, text_obj=None, verbose=None, enable_function_calls=None, chat=True,
                          token_prob_delta=None, token_prob_abs=None, handle_functions=True, stop=None, **kwargs):
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
                self.add_tracker_entry(ConversationRoles.USER, content=text_obj)
            prompt_obj = self.build_prompt()
            self.prompt = prompt_obj
            if self.settings.prompt_obj_is_str:
                stop.append(self.symbols["USER"])
                stop.append(self.symbols["ASSISTANT"])
            ret_text = self.create_native_completion(prompt_obj, **add_kwargs, **kwargs)  # return_factory(prompt_obj)

        elif text_obj and self.settings.prompt_obj_is_str:
            ret_text = self.create_native_completion(text_obj, **add_kwargs, **kwargs)

        if not chat and not text_obj:
            raise Exception("No prompt given!")

        self.raw_generated_text += ret_text
        if not enable_function_calls:
            if chat:
                self.add_tracker_entry(ConversationRoles.ASSISTANT, content=ret_text)
            end = timer()
            self.finish_meta["total_finish_time"] = end - start
            return self.raw_generated_text

        status, seq, new_text = self._extract_and_handle_functions(ret_text, call_functions=handle_functions)

        if status != ParseStatus.NO_FUNC_SEQUENCE_FOUND:
            self.finish_meta["function_call"]["found"] = True
            self.finish_meta["function_call"]["sequence"] = seq

        if status != ParseStatus.PARSED_EXECUTED_OK:
            if chat:
                self.add_tracker_entry(ConversationRoles.ASSISTANT, content=ret_text)
            end = timer()
            self.finish_meta["total_finish_time"] = end - start
            return self.raw_generated_text
        self.raw_generated_text = new_text + "\n"
        try:
            stop.remove(self.settings.function_sequence[1])
        except:
            pass
        prompt_obj = self.build_prompt()
        ret_text = self.create_native_completion(prompt_obj, **add_kwargs, **kwargs)

        self.raw_generated_text += ret_text

        if chat:
            self.add_tracker_entry(ConversationRoles.ASSISTANT, content=ret_text)
        end = timer()
        self.finish_meta["total_finish_time"] = end - start
        return self.raw_generated_text

    def _extract_and_handle_functions(self, text, call_functions=True):
        """
        Extract function sequences from text and parse and potentially execute them

        :param text:
        :param call_functions: Call or just collect calls and return
        :return:
        """
        start_seq = self.settings.function_sequence[1]
        end_seq = self.settings.function_sequence[1]
        pattern = re.escape(start_seq) + '(.*?)' + re.escape(end_seq)
        matches = [(m.group(1), m.span()) for m in re.finditer(pattern, text, re.DOTALL)]

        if len(matches) == 0:
            text += end_seq
            pattern = re.escape(start_seq) + '(.*?)' + re.escape(end_seq)
            matches = [(m.group(1), m.span(), m) for m in re.finditer(pattern, text, re.DOTALL)]

            if len(matches) == 0:
                self.finish_meta["parse_status"] = ParseStatus.NO_FUNC_SEQUENCE_FOUND
                return ParseStatus.NO_FUNC_SEQUENCE_FOUND, None, None
        func_seq = matches[0][0].strip()
        try:
            ast_obj = ast.parse(func_seq)
        except Exception as e:
            if call_functions:
                raise Exception("Models doing stuff wrong is not yet correctly handled")
            self.finish_meta["parse_status"] = ParseStatus.UNPARSEABLE_FUNC_FOUND
            return ParseStatus.UNPARSEABLE_FUNC_FOUND, func_seq, e

        if not call_functions:
            visitor = python_parsing.CodeVisitor(collect=True)
            visitor.visit(ast_obj)
            self.finish_meta["parse_status"] = ParseStatus.PARSED_DICT_RETURN
            return ParseStatus.PARSED_DICT_RETURN, func_seq, visitor.collection

        visitor = python_parsing.CodeVisitor(self.available_functions)
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
        self.finish_meta["parse_status"] = ParseStatus.PARSED_EXECUTED_OK

        return ParseStatus.PARSED_EXECUTED_OK, func_seq, text[:loc]

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
                self.add_tracker_entry(ConversationRoles.USER, content=text_obj)
            prompt_obj = self.build_prompt()
            self.prompt = prompt_obj
            if self.settings.prompt_obj_is_str:
                stop.append(self.symbols["USER"])
                stop.append(self.symbols["ASSISTANT"])
            token_generator = self.create_native_generator(prompt_obj, **add_kwargs,
                                                           **kwargs)  # return_factory(prompt_obj)

        elif text_obj and self.settings.prompt_obj_is_str:
            token_generator = self.create_native_generator(text_obj, **add_kwargs, **kwargs)

        if not chat and not text_obj:
            raise Exception("No prompt given!")

        sequences = []
        for i, x in self.settings.preserved_sequences.items():
            x["type"] = i
            sequences.append(x)
        if not chat and enable_function_calls:
            sequences.append({"type": "function_call", "start": self.settings.function_sequence[0],
                              "end": self.settings.function_sequence[1]})

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
                        if len(buffer) != 0:
                            sequence_tokens = []
                            yield "".join(buf_temp), None, buf_logits_temp
                    else:
                        status, seq, new_text = self._extract_and_handle_functions(self.raw_generated_text,
                                                                                   call_functions=handle_functions)
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

    def build_prompt_as_str(self, new_lines_per_role=1, new_lines_afer_role=0, block_gen_prefix=False, raw=False):
        """
        Build a prompt in string form
        :param new_lines_per_role: Newlines after Role+message
        :param new_lines_afer_role: Newlines after role
        :param block_gen_prefix: Whether to add generation prefix. Normally this leads a prompt to end with e.g.
        Assistant: so that the model does not continue to write as a user.
        :param raw: If true don't resolve symbols
        :return: str
        """

        def rep_sym(str, entry=None):
            return str if raw else self.replace_symbols(str, entry)

        prompt = ""
        after_role = "\n" * new_lines_afer_role if new_lines_afer_role != 0 else " "
        if self.conversation_history.system_message and self.conversation_history.system_message != "":
            prompt += f"{self.symbols['SYSTEM']}:{after_role}{rep_sym(self.system_msg)}" + "\n" * new_lines_per_role

        for i in self.conversation_history.tracker:
            role = self.symbols[str(i["role"])]
            prompt += f"{role}:{after_role}{rep_sym(i['content'], i)}" + "\n" * new_lines_per_role
        if block_gen_prefix:
            prompt = prompt[:-1]
        if not block_gen_prefix and self.settings.generation_prefix and self.settings.generation_prefix != "":
            prompt += rep_sym(self.settings.generation_prefix)
        return prompt


def _get_enum_value(input_value, enum_type):
    if isinstance(input_value, str):
        try:
            return enum_type[input_value.upper()]
        except KeyError:
            raise ValueError(f"'{input_value}' not found in {enum_type.__name__}.")
    elif isinstance(input_value, enum_type):
        return input_value
    else:
        raise TypeError(f"Invalid input type. Expected enum value or string, got {type(input_value).__name__}.")
