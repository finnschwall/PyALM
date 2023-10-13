from abc import abstractmethod
import psutil
import ast
import enum
import re
from timeit import default_timer as timer
from pylot import python_parsing
from typing import Type
from functools import partial

class ConversationRoles(enum.Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"

    def __str__(self) -> str:
        return self.value


class FunctionFormat(enum.Enum):
    PYDOC = "PYDOC"
    JSON = "JSON"
    MODEL_SPECIFIC = "MODEL_SPECIFIC"

    def __str__(self) -> str:
        return self.value


class ALM:
    def __init__(self, model_path_or_name, n_ctx=2048, verbose=0):
        self.model = model_path_or_name
        self.verbose = verbose
        self.atomic_sequences = {"functions": ["+++", "---"],  # ["➡️", "⬅️"],["❬", "❭"]
                                 "latex_double": ["$$", "$$"],
                                 "latex_single": ["$", "$"]}
        func_inclusion_message = """[[LIST_OF_FUNCTIONS]]
Above you is a list of functions you can call. To call them enclose them with [[FUNC_DELIMITER_START]] and end the call with [[FUNC_DELIMITER_END]].
This
[[FUNC_DELIMITER_START]]foo(bar=3)[[FUNC_DELIMITER_END]]
would call the function foo with bar=3. The function(s) will return immediately. The values will be in the inverse sequence of the function enclosement.  
You can only call the functions listed.
You can and HAVE TO call functions during the text response not in a a separate response!
Before you call a function please inform the user so he is aware of possible waiting times.
"""
        self.symbols = {"FUNC_DELIMITER_START": self.atomic_sequences["functions"][0],
                        "FUNC_DELIMITER_END": self.atomic_sequences["functions"][1],
                        "ASSISTANT": "Assistant", "USER": "User", "SYSTEM": "System",
                        "FUNC_INCLUSION_MESSAGE": func_inclusion_message, "LIST_OF_FUNCTIONS": "No functions available"}
        self.base_settings = {"GENERATION_PREFIX": "[[ASSISTANT]]: ", "FUNCTIONS_ENABLED": True,
                              "FUNCTION_AUTOINTEGRATION": True,
                              "preserved_sequences": [{"start": self.atomic_sequences["functions"][0],
                                                       "end": self.atomic_sequences["functions"][1],
                                                       "is_function": True, "type": "function"},
                                                      {"start": "$$", "end": "$$", "name": "latex1"}]}
        self.settings = dict(self.base_settings)
        self.func_map = {}

        self.available_functions = []
        self.conv_history = []

        self.system_msg = {}
        self.generated_text = ""

        self.prompt_text_is_str = False

    def adopt_from_alm(self, other: Type['ALM']):
        self.conv_history = other.conv_history
        self.system_msg = other.system_msg
        self.verbose = other.verbose
        self.atomic_sequences = other.atomic_sequences

    def _repl(self, match):
        if match[1] in self.symbols:
            return self.symbols[match[1]]
        return match[1]

    def _replace_symbols(self, text):
        pattern = r'\[\[(.*?)\]\]'
        matches = list(re.finditer(pattern, text))
        text = re.sub(pattern, self._repl, text)
        return text

    def save_history(self, path):
        with open(path, "w") as f:
            f.write(yaml.dump(self.conv_history))

    def load_history(self, path):
        with open(path, "r") as f:
            self.conv_history = yaml.full_load(f.read())

    def set_system_message(self, msg, prepend_function_support=True):
        if prepend_function_support:
            msg = self.symbols["FUNC_INCLUSION_MESSAGE"] + msg
        self.system_msg["content"] = msg

    @abstractmethod
    def tokenize(self, text):
        raise NotImplementedError()

    @abstractmethod
    def tokenize_as_str(self, text):
        raise NotImplementedError()

    @abstractmethod
    def get_n_tokens(self, text):
        raise NotImplementedError()

    @abstractmethod
    def create_native_generator(self, text, token_prob_delta=None, endless=False,
                                token_prob_abs=None, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def build_prompt(self, preserve_flow=False):
        raise NotImplementedError()

    def reset_tracker(self):
        self.conv_history = []

    def add_tracker_entry(self, role, content=None, meta=None, function_calls=None, context=None, feedback=None,
                          sentiment=None, add_keys=None):

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
                                    return_function_call=False, token_prob_delta=None, token_prob_abs=None,
                          handle_functions= True, **kwargs):
        self.generated_text=""
        if not self.prompt_text_is_str:
            chat = True
        if enable_function_calls is None:
            enable_function_calls = self.settings["FUNCTIONS_ENABLED"]

        start = timer()
        if not verbose:
            verbose = self.verbose

        if not "stop" in kwargs:
            stop = []
        elif isinstance("stop", str):
            stop = [stop]
        if not enable_function_calls or not chat:
            try:
                stop.remove(self.atomic_sequences["functions"][1])
            except:
                pass

        # if enable_function_calls:
        #     stop.append(self.atomic_sequences["functions"][1])
        #     print(stop)
        return_factory = partial(self.create_native_generator, stop=stop, token_prob_delta=token_prob_delta,
                                                           token_prob_abs=token_prob_abs, stream=False, **kwargs)
        if chat:
            if text_obj:
                self.add_tracker_entry(ConversationRoles.USER, content=text_obj)
            prompt_obj = self.build_prompt()
            self.prompt = prompt_obj
            if self.prompt_text_is_str:
                stop.append(self.symbols["USER"])
                stop.append(self.symbols["ASSISTANT"])
            ret_text =  return_factory(prompt_obj)
                # self.create_native_generator(prompt_obj, stop=stop, token_prob_delta=token_prob_delta,
                #                                            token_prob_abs=token_prob_abs, stream=False, **kwargs)
        if text_obj and not chat and self.prompt_text_is_str:
            ret_text = return_factory(text_obj)
                # self.create_native_generator(text_obj, stop=stop, token_prob_delta=token_prob_delta,
                #                                            token_prob_abs=token_prob_abs, stream=False, **kwargs)
        if not chat and not text_obj:
            raise Exception("No prompt given!")

        pattern = re.escape(self.atomic_sequences["functions"][0]) + '(.*?)' + re.escape(self.atomic_sequences["functions"][1])
        # matches = re.findall(pattern, ret_text, re.DOTALL)
        matches = [(m.group(1),m.span()) for m in re.finditer(pattern, ret_text, re.DOTALL)]
        if handle_functions and len(matches)!=0:
            func_seq = matches[0][0].strip()
            visitor = python_parsing.CodeVisitor(self.func_map)
            visitor.visit(ast.parse(func_seq))
            if "__call_res__" in visitor.variables:
                ret_val = visitor.variables["__call_res__"]
                new_assistant_text = ret_text + "\n" + self.atomic_sequences["functions"][1] + \
                                     ret_val + self.atomic_sequences["functions"][0]
                self.add_tracker_entry(ConversationRoles.ASSISTANT, content=new_assistant_text,
                                       function_calls={"original_call": func_seq, "return": ret_val})
                prompt_obj = self.build_prompt()
                try:
                    stop.remove(self.atomic_sequences["functions"][1])
                except:
                    pass
                self.generated_text += ret_text[:matches[0][1][0]] +"\n"
                ret_text = return_factory(prompt_obj)


        self.generated_text += ret_text
        if chat:
            self.add_tracker_entry(ConversationRoles.ASSISTANT, content=ret_text)
        end = timer()
        self.finish_meta["total_finish_time"] = end - start
        return self.generated_text


    # this could be improved by using raw token numbers. Then comparison to token number would be possible. would remedy whitespace issue
    # but that would break closed source compatability
    def create_completion_generator(self, text_obj=None, verbose=None, enable_function_calls=None, chat=True,
                                    return_function_call=False, token_prob_delta=None, token_prob_abs=None,
                                    **kwargs):
        if not self.prompt_text_is_str:
            chat = True
        if enable_function_calls is None:
            enable_function_calls = self.settings["FUNCTIONS_ENABLED"]

        start = timer()
        if not verbose:
            verbose = self.verbose

        if not "stop" in kwargs:
            stop = []
        elif isinstance("stop", str):
            stop = [stop]
        if not enable_function_calls or not chat:
            try:
                stop.remove(self.atomic_sequences["functions"][1])
            except:
                pass

        # if enable_function_calls:
        #     stop.append(self.atomic_sequences["functions"][1])
        #     print(stop)

        if chat:
            if text_obj:
                self.add_tracker_entry(ConversationRoles.USER, content=text_obj)
            prompt_obj = self.build_prompt()
            self.prompt = prompt_obj
            if self.prompt_text_is_str:
                stop.append(self.symbols["USER"])
                stop.append(self.symbols["ASSISTANT"])
            token_generator = self.create_native_generator(prompt_obj, stop=stop, token_prob_delta=token_prob_delta,
                                                           token_prob_abs=token_prob_abs, **kwargs)
        if text_obj and not chat and self.prompt_text_is_str:
            token_generator = self.create_native_generator(text_obj, stop=stop, token_prob_delta=token_prob_delta,
                                                           token_prob_abs=token_prob_abs, **kwargs)
        if not chat and not text_obj:
            raise Exception("No prompt given!")

        self.generated_text = ""

        sequences = self.settings["preserved_sequences"]
        if not chat:
            sequences = [d for d in sequences if d.get('type') != 'function']

        buffer = []
        buffer_logits = []
        sequence_tokens = []
        start_sequence = None
        end_sequence = None
        in_sequence = False
        yield_type = "token"

        # cleanup_sentinel = object()
        # def token_generator_with_cleanup():
        #     yield from token_generator
        #     yield cleanup_sentinel
        generator_list = [token_generator]

        def token_generator_with_insertions():
            while True:
                if len(generator_list) != 0:
                    cur_gen = generator_list.pop()
                    yield from cur_gen
                else:
                    break

        for token, logits_or_none in token_generator_with_insertions():
            # if token is cleanup_sentinel:
            #     yield "".join(buffer), None, None
            #     break
            # print(token, end="")
            self.generated_text += token
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
                    ret_val = None
                    if yield_type == "function":
                        # print("\n---")
                        func_seq_full = "".join(sequence_tokens).strip()
                        func_seq = func_seq_full[len(start_sequence):-len(end_sequence)]
                        visitor = python_parsing.CodeVisitor(self.func_map)
                        visitor.visit(ast.parse(func_seq))
                        if "__call_res__" in visitor.variables:
                            # print("\n-----------------------------------------------")
                            print(generator_list)
                            if len(generator_list)!=0:
                                rem_tok_gen = generator_list[0]
                                for discard in rem_tok_gen:
                                    pass
                            ret_val = visitor.variables["__call_res__"]
                            # code_insertion = f"#Called function\n{func_seq}\n#Return value\n{ret_val}"
                            # code_loc = self.generated_text.find(func_seq_full)
                            new_assistant_text = self.generated_text + "\n" + self.atomic_sequences["functions"][1] + \
                                                 ret_val + self.atomic_sequences["functions"][0]
                            self.add_tracker_entry(ConversationRoles.ASSISTANT, content=new_assistant_text,
                                                   function_calls={"original_call": func_seq, "return": ret_val})

                            prompt_obj = self.build_prompt()  # preserve_flow=True)
                            try:
                                stop.remove(self.atomic_sequences["functions"][1])
                            except:
                                pass
                            sequences = [d for d in sequences if d.get('type') != 'function']
                            token_generator_funcs = self.create_native_generator(prompt_obj, stop=stop, **kwargs)
                            generator_list.append(token_generator_funcs)
                            self.generated_text = ""
                            # yield "\n", None, None
                    else:
                        yield ''.join(sequence_tokens), yield_type, logits_or_none
                    yield_type = "token"
                    sequence_tokens = []
                    buffer = []
                    buffer_logits = []

        if len(buffer) != 0:
            yield "".join(buffer), None, buffer_logits
        if chat:
            self.add_tracker_entry(ConversationRoles.ASSISTANT, content=self.generated_text)
        end = timer()
        self.finish_meta["total_finish_time"] = end - start

    def build_prompt_as_str(self, new_lines_per_role=1, new_lines_afer_role=0, block_gen_prefix=False):
        prompt = ""
        after_role = "\n" * new_lines_afer_role if new_lines_afer_role != 0 else " "
        if "content" in self.system_msg:
            prompt += f"{self.symbols['SYSTEM']}:{after_role}{self.system_msg['content']}" + "\n" * new_lines_per_role

        for i in self.conv_history:
            role = self.symbols[str(i["role"])]
            prompt += f"{role}:{after_role}{i['content']}" + "\n" * new_lines_per_role
        if block_gen_prefix:
            prompt = prompt[:-1]
        if not block_gen_prefix and "GENERATION_PREFIX" in self.settings and self.settings["GENERATION_PREFIX"] != "":
            prompt += self.settings["GENERATION_PREFIX"]
        prompt = self._replace_symbols(prompt)
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
