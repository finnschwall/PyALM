from abc import abstractmethod
import psutil
import ast
import enum
import re
from timeit import default_timer as timer
from pylot import python_parsing
class ConversationRoles(enum.Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    
    def __str__(self) -> str:
        return self.value



class ALM:
    def __init__(self, model_path_or_name, n_ctx=2048, verbose=0):
        self.model = model_path_or_name
        self.verbose=verbose
        self.atomic_sequences = {"functions": ["➡️", "⬅️"],
                                 "latex_double" : ["$$","$$"],
                                "latex_single" : ["$","$"]}
        self.symbols = {"FUNC_DELIMITER_START":self.atomic_sequences["functions"][0], "FUNC_DELIMITER_END":self.atomic_sequences["functions"][1],
                       "ASSISTANT": "Assistant", "USER":"User"}
        self.base_settings = {"GENERATION_PREFIX": "[[ASSISTANT]]: "}
        self.settings = dict(self.base_settings)

        self.conv_history= []

        self.system_msg = None

        self.prompt_text_is_str = False

    def _repl(self, match):
        if match[1] in self.symbols:
            return self.symbols[match[1]]
        return match[1]

    def replace_symbols(self, text):
        pattern = r'\[\[(.*?)\]\]'
        matches = list(re.finditer(pattern, text))
        text= re.sub(pattern, self._repl, text)
        return text

    def save_history(self, path):
        with open(path,"w") as f:
            f.write(yaml.dump(self.conv_history))

    def load_history(self, path):
        with open(path,"r") as f:
            self.conv_history = yaml.full_load(f.read())
    
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
    def create_native_generator(self, text, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def build_prompt(self):
        raise NotImplementedError()

    def reset_tracker(self):
        self.conv_history = []

    def add_tracker_entry(self, role, content= None, meta=None, function_calls=None, context = None, feedback=None, sentiment= None, add_keys = None):

        role = _get_enum_value(role, ConversationRoles)
        
        msg = {"role" : role}
        excl = ["msg", "role", "add_keys", "excl", "loc", "self"]
        loc = locals()
        for i in loc:
            if i in excl:
                continue
            if loc[i]:
                msg[i] = loc[i]
        if add_keys:
            msg = msg | add_keys
            msg = {"role" : role}
        self.conv_history.append(msg)

    @staticmethod
    def functions_to_dict(functions):
        if not isinstance(functions, list):
            functions = [functions]
        dic_list = []
        for i in functions:
            func_as_dic = python_parsing.function_signature_to_dict(i)
            dic_list.append(func_as_dic)
        return dic_list

    def register_functions(self, functions):
        pass

    
        


    # this could be improved by using raw token numbers. Then comparison to token number would be possible. would remedy whitespace issue
    # but that would break closed source compatability
    def create_completion_generator(self,text_obj=None, verbose=None, enable_function_calls = True,
                          preserved_sequences =  [{"start":"++","end":"--","is_function":True, "name":"function"}, {"start":"$$","end":"$$"}], chat=False,**kwargs):
        
        start = timer()
        if not verbose:
            verbose=self.verbose

        if not "stop" in kwargs:
            stop = []
        elif isinstance("stop",str):
            stop = [stop]
        
        if chat or not self.prompt_text_is_str:
            if text_obj:
                self.add_tracker_entry(ConversationRoles.USER, content=text_obj)
            prompt_obj = self.build_prompt()
            self.prompt = prompt_obj
            if self.prompt_text_is_str:
                stop.append(self.symbols["USER"])
            token_generator = self.create_native_generator(prompt_obj, stop=stop,**kwargs)
        if text_obj and not chat and self.prompt_text_is_str:
            token_generator = self.create_native_generator(text_obj, stop, **kwargs)
            
        self.generated_text = ""
        
        
        sequences = preserved_sequences

        buffer = []
        sequence_tokens = []
        start_sequence = None
        end_sequence = None
        in_sequence = False
        yield_type = "token"
    
        for token in token_generator:
            self.generated_text+=token
            buffer.append(token)
            buffer_str = ''.join(buffer)
    
            if not in_sequence:
                for sequence in sequences:
                    if buffer_str.strip().startswith(sequence['start']):
                        in_sequence = True
                        start_sequence = sequence['start']
                        end_sequence = sequence['end']
                        sequence_tokens.extend(buffer)
                        yield_type = start_sequence + "XYZ" +end_sequence if not "type" in sequence else sequence["type"]
                        buffer = []
                        break
                if not in_sequence and len(buffer) > len(max(sequences, key=lambda s: len(s['start']))['start']):
                    yield buffer.pop(0), yield_type, None
            else:

                sequence_tokens.append(token)
                if buffer_str.endswith(end_sequence):
                    in_sequence = False
                    ret_val = None
                    if yield_type=="function":
                        pass
                    yield ''.join(sequence_tokens), yield_type, ret_val
                    yield_type="token"
                    sequence_tokens = []
                    buffer = []
        if chat:
            self.add_tracker_entry(ConversationRoles.ASSISTANT, content = self.generated_text)
        end = timer()
        self.finish_meta["total_finish_time"] = end - start
        # print("lol")


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
    




class CodeVisitor(ast.NodeVisitor):
    def __init__(self, func_map):
        self.func_map = func_map
        self.variables = {}

    def visit_Call(self, node):
        func_name = node.func.id
        args = [self.variables.get(arg.id, None) if isinstance(arg, ast.Name) else ast.literal_eval(arg) for arg in node.args]
        kwargs = {kw.arg: self.variables.get(kw.value.id, None) if isinstance(kw.value, ast.Name) else ast.literal_eval(kw.value) for kw in node.keywords}

        # print(f'Function call: {func_name}({args}, {kwargs})')

        resolved_func = self.func_map.get(func_name)
        if resolved_func:
            result = resolved_func(*args, **kwargs)
            self.variables['__call_res__'] = result
        return result

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if isinstance(node.value, ast.Call):
                value = self.visit(node.value)
            elif isinstance(node.value, ast.Name):
                value = self.variables.get(node.value.id)
            else:
                value = ast.literal_eval(node.value)

            self.variables[var_name] = value
            # print(f'Variable assignment: {var_name} = {value}')


# def test(x,y):
#     print(f'Plotting {x} {y}')
#     return 5
# func_map = {'plot': test, 'show': lambda: print('Showing plot')}

# visitor = CodeVisitor(func_map)
# visitor.visit(ast.parse('result = plot("sin(x)^2", "green")\nshow()'))
# visitor.variables