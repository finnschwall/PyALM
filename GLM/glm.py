from abc import abstractmethod
import psutil
import ast
class GLM:
    def __init__(self, model_path_or_name, n_ctx=2048, verbose=0):
        self.model = model_path_or_name
        self.verbose=verbose
        self.atomic_sequences = {"functions": ["➡️", "⬅️"],
                                 "latex_double" : ["$$","$$"],
                                "latex_single" : ["$","$"]}
        self.symbols = {"FUNC_DELIMITER_START":self.atomic_sequences["functions"][0], "FUNC_DELIMITER_END":self.atomic_sequences["functions"][1]}

        self.conv_history= {}

        self.system_msg = None

    def _repl(self, match):
        return match[1]

    def replace_symbols(self, text):
        pattern = r'\[\[(.*?)\]\]'
        matches = list(re.finditer(pattern, txt))
        re.sub(pattern, self.repl, txt)
        return txt

    def save_history(self, path):
        with open(path,"w") as f:
            f.write(yaml.dump(self.conv_history))
    
    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def tokenize_as_str(self, text):
        pass

    @abstractmethod
    def get_n_tokens(self, text):
        pass

    @abstractmethod
    def create_native_generator(self, text, *args, **kwargs):
        raise NotImplementedException()

    @abstractmethod
    def build_model_input_from_history(self, yaml_inp):
        pass


        # this could be improved by using raw token numbers. Then comparison to token number would be possible. would remedy whitespace issue
    # but that would break closed source compatability
    def create_completion(self,text, verbose=None, max_tokens=256, enable_function_calls = True,
                          preserved_sequences =  [{"start":"++","end":"--","is_function":True, "name":"function"}, {"start":"$$","end":"$$"}],**kwargs):
        if not verbose:
            verbose=self.verbose
        #for LLaMA=max tokens -1: shift context, -2 stop when full

        token_generator = self.create_native_generator(text,**kwargs)

        self.generated_text = ""
        self.prompt = text

        
        sequences = preserved_sequences

        buffer = []
        sequence_tokens = []
        start_sequence = None
        end_sequence = None
        in_sequence = False
        yield_type = "token"
    
        for token in token_generator:
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



    


class CodeVisitor(ast.NodeVisitor):
    def __init__(self, func_map):
        self.func_map = func_map
        self.variables = {}

    def visit_Call(self, node):
        # Get function name
        func_name = node.func.id

        # Get arguments
        args = [self.variables.get(arg.id, None) if isinstance(arg, ast.Name) else ast.literal_eval(arg) for arg in node.args]

        # Get keyword arguments
        kwargs = {kw.arg: self.variables.get(kw.value.id, None) if isinstance(kw.value, ast.Name) else ast.literal_eval(kw.value) for kw in node.keywords}

        print(f'Function call: {func_name}({args}, {kwargs})')

        resolved_func = self.func_map.get(func_name)
        if resolved_func:
            result = resolved_func(*args, **kwargs)   # This executes the function call.
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
            print(f'Variable assignment: {var_name} = {value}')


# def test(x,y):
#     print(f'Plotting {x} {y}')
#     return 5
# func_map = {'plot': test, 'show': lambda: print('Showing plot')}

# visitor = CodeVisitor(func_map)
# visitor.visit(ast.parse('result = plot("sin(x)^2", "green")\nshow()'))
# visitor.variables