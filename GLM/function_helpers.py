import inspect
from docstring_parser import parse

def generate_python_doc(func_name, func_dict):
    args = func_dict.get('args', [])
    kwargs = func_dict.get('kwargs', [])
    doc = f"def {func_name}("
    for arg in args:
        arg_type = ""
        if "type" in arg:
            arg_type = ":"+arg["type"]
        doc += f"{arg['name']}"+ arg_type+", "
    for kwarg in kwargs:
        kwarg_type = ""
        if "type" in kwarg:
            kwarg_type = ":"+kwarg["type"]
        doc += f"{kwarg['name']}{kwarg_type}={kwarg.get('default', 'None')}, "
    doc = doc.rstrip(', ') + ")\n"
    doc += '"""\n'
    if "description" in func_dict:
        doc+=func_dict["description"]+"\n"
    for arg in args:
        if "description" in arg:
            doc += f":param {arg['name']}: {arg.get('description', '')}\n"
    for kwarg in kwargs:
        doc += f":param {kwarg['name']}: {kwarg.get('description', '')}\n"
    if "return" in func_dict:
        doc += ":return: "+func_dict["return"]
    doc += '"""'
    return doc


def function_signature_to_dict(func):
    sig = inspect.signature(func)
    params = sig.parameters

    doc = parse(func.__doc__)

    args = []
    kwargs = []

    for name, param in params.items():
        if param.default == inspect.Parameter.empty:
            arg = {'name': name}
            for doc_param in doc.params:
                if doc_param.arg_name == name:
                    print(doc_param.type_name)
                    if doc_param.type_name:
                        arg['type'] = doc_param.type_name
                    if doc_param.description and doc_param.description != "":
                        arg['description'] = doc_param.description
            args.append(arg)
        else:
            kwarg = {'name': name, 'default': param.default}
            for doc_param in doc.params:
                if doc_param.arg_name == name:
                    kwarg_type = doc_param.type_name
                    if not kwarg_type:
                        if param.default:
                            kwarg_type = type(param.default).__name__
                    if kwarg_type:
                        kwarg['type'] = kwarg_type
                    if doc_param.description and doc_param.description != "":
                        kwarg['description'] = doc_param.description
            kwargs.append(kwarg)

    return {
        'name': func.__name__,
        'description': doc.short_description,
        'args': args,
        'kwargs': kwargs
    }