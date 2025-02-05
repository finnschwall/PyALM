import dataclasses as dc
import enum
import yaml
from abc import ABC


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


class ConversationRoles(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"

    def __str__(self) -> str:
        return self.value

@dc.dataclass
class DataYAML(ABC):

    def to_dict(self):
        return dc.asdict(self)

    @classmethod
    def from_dict(cls, dict_obj):
        return cls(**dict_obj)

    def to_yaml(self):
        yaml_str = yaml.dump(self.to_dict(), sort_keys=False)
        return yaml_str

    @classmethod
    def from_yaml(cls, text, ):
        data = text
        data = yaml.full_load(data)
        if type(data) is not dict:
            raise Exception("Input is valid YAML but not valid data")
        instance = cls.from_dict(data)
        return instance



# glob_inv_scheme =  {"USER": ConversationRoles.ASSISTANT, "ASSISTANT": ConversationRoles.USER}
glob_inv_scheme = {ConversationRoles.USER: ConversationRoles.ASSISTANT,
                   ConversationRoles.ASSISTANT: ConversationRoles.USER,
                   "user": "assistant", "assistant": "user"}


@dc.dataclass(kw_only=True)
class ConversationTracker(DataYAML):
    system_message: str = None
    tracker: list = dc.field(default_factory=list)
    metadata: dict = dc.field(default_factory=dict)
    data: dict = dc.field(default_factory=dict)
    user_info = None

    def reset_tracker(self):
        temp = self.tracker
        self.tracker = []
        return temp

    @property
    def inversion_scheme(self):
        global glob_inv_scheme
        if self.data.get("inversion_scheme"):
            return self.data["inversion_scheme"]
        return glob_inv_scheme

    @inversion_scheme.setter
    def inversion_scheme(self, value):
        self.data["inversion_scheme"] = value

    def invert_roles(self, inversion_scheme=None):
        global glob_inv_scheme
        if inversion_scheme is None:
            inversion_scheme = glob_inv_scheme  # self.inversion_scheme
        for i, x in enumerate(self.tracker):
            self.tracker[i]["role"] = inversion_scheme.get(self.tracker[i]["role"], self.tracker[i]["role"])
        if "system_message2" in self.data:
            orig_system = self.system_message
            self.system_message = self.data["system_message2"]
            self.data["system_message2"] = orig_system

    def __getitem__(self, item):
        return self.tracker[item]

    def __setitem__(self, key, value):
        self.tracker[key] = value

    def __len__(self):
        return len(self.tracker)

    def get_last_message(self, role=None, include_depth=False):
        if not isinstance(role, str):
            role = str(role)
        for i in range(len(self.tracker) - 1, -1, -1):
            if not role:
                if include_depth:
                    return self.tracker[i], i
                return self.tracker[i]
            if self.tracker[i]["role"] == role:
                if include_depth:
                    return self.tracker[i], i
                return self.tracker[i]
        if include_depth:
            return None, -1
        return None

    def get_last_entries(self):
        if len(self.tracker) == 0:
            return []
        if len(self.tracker) == 1:
            return self.tracker[1]
        last_role = self.tracker[-1]["role"]

        index = -1
        inverted_role = self.inversion_scheme.get(last_role)
        for i in range(len(self.tracker) - 1, -1, -1):
            if self.tracker[i]["role"] == inverted_role:
                index = i
                break
        if index == -1:
            return self.tracker
        ret = []
        for i in range(len(self.tracker) - 1, index, -1):
            ret.append(self.tracker[i])
        return ret

    def pop_entry(self):
        if len(self.tracker) == 0:
            return []
        if len(self.tracker) == 1:
            return self.tracker.pop(1)
        last_role = self.tracker[-1]["role"]

        index = -1
        inverted_role = self.inversion_scheme.get(last_role)
        for i in range(len(self.tracker) - 1, -1, -1):
            if self.tracker[i]["role"] == inverted_role:
                index = i
                break
        if index == -1:
            return self.tracker.pop()
        ret = []
        for i in range(len(self.tracker) - 1, index, -1):
            ret.append(self.tracker.pop(i))
        return ret

    def add_entry(self, content=None, role=None, metadata=None, code=None, return_value=None, feedback=None,
                  sentiment=None,processing=None, add_keys=None):
        if not isinstance(role, str):
            role = str(role)
        if not role and len(self.tracker) == 0:
            role = str(ConversationRoles.USER)
        elif not role:
            role = self.inversion_scheme.get(self.tracker[-1]["role"])
        if not content and not code:
            raise ValueError("Either content or code must be provided.")
        loc_dic = locals()
        del loc_dic["self"]
        del loc_dic["role"]
        self._add_entry(role, **loc_dic)

    def _add_entry(self, role, content=None, metadata=None, feedback=None, code=None, return_value=None,
                   sentiment=None, processing=None, add_keys=None):
        # role = _get_enum_value(role, ConversationRoles)

        entry = {"role": role}
        if content:
            entry["content"] = content
        if metadata:
            entry["metadata"] = metadata
        if code:
            entry["code"] = code
        if return_value:
            entry["return_value"] = return_value
        if feedback:
            entry["feedback"] = feedback
        if processing:
            entry["processing"] = processing
        if add_keys:
            entry = entry | add_keys
        entry["index"] = 0 if len(self.tracker) == 0 else self.tracker[-1]["index"] + 1
        self.tracker.append(entry)
        return entry


@dc.dataclass(kw_only=True)
class ALMSettings(DataYAML):
    verbose: int = 0
    preserved_sequences: dict = dc.field(
        default_factory=lambda: {"latex_double": {"start": "$$", "end": "$$", "name": "latex_double_dollar"}})
    function_sequence: tuple = dc.field(default_factory=lambda: ("$$$CODE_START", "$$$CODE_END"))
    to_user_sequence: str = "$$$TO_USER"
    global_enable_function_calls: bool = False
    automatic_function_integration: bool = False
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
"""
    prompt_obj_is_str: bool = True
    include_conv_id_as_stop = True


def _data_yaml_representer(dumper, data):
    return dumper.represent_dict({'class': type(data).__name__, 'data': data.to_dict()})


def _data_yaml_constructor(loader, node):
    data = loader.construct_dict(node)
    cls = globals()[data['class']]
    return cls.from_dict(data['data'])


# for i in [DataYAML, ConversationTracker]:
#     yaml.add_representer(i, _data_yaml_representer)
#     yaml.add_constructor('!' + i.__name__, _data_yaml_constructor)


def conversation_role_representer(dumper, data):
    return dumper.represent_scalar('!ConversationRole', str(data))


def conversation_role_constructor(loader, node):
    value = loader.construct_scalar(node)
    return _get_enum_value(value, ConversationRoles)


# yaml.add_representer(ConversationRoles, conversation_role_representer)
# yaml.add_constructor('!ConversationRole', conversation_role_constructor)

def enum_representer(dumper, data):
    """Convert enum to string when dumping YAML"""
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

def enum_constructor(loader, node):
    """Convert string back to enum when loading YAML"""
    value = loader.construct_scalar(node)
    return type(node.tag)(value)

# Add representers for your specific enum
# yaml.add_representer(ConversationRoles, enum_representer)
# yaml.add_constructor('!ConversationRoles', enum_constructor)