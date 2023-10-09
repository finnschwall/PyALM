# WARNING
This readme being somewhat intelligible does not mean the project is ready to be installed!

(it isn't)

# Usage
## LLaMa

If you don't have one: Download a model. Good adress is e.g. [TheBloke](https://huggingface.co/TheBloke).
On CPU only I recommend 7B models. 13B only if you are really patient.

In the [C++ library](https://github.com/ggerganov/llama.cpp) there are various scripts that can be used to obtain more models e.g. by quantizing models. Although that does require considerable resources.
### Chat
```python
from GLM import LLaMa, ConversationRoles as cr
llm = LLaMa(PATH_TO_MODEL)

llm.add_tracker_entry(cr.USER, "Whats up?")
llm.add_tracker_entry(cr.ASSISTANT, "just chillin WBU?")
llm.add_tracker_entry(cr.USER, "Utilizing some python stuff for abstracting large language models in an end user centered way")

for i in llm.create_completion_generator(chat=True, max_tokens=250):
    print(i[0], end="")
```

### Raw text

With LLaMa raw input into the model is also possible
```python
from GLM import LLaMa
llm = LLaMa(PATH_TO_MODEL)

for i in llm.create_completion_generator("O Captain! my Captain! our fearful trip", max_tokens=250):
    print(i[0], end="")
```

### Hardware acceleration

For utilizing the GPU install GPU support and use the `n_layers` parameter.

```python
from GLM import LLaMa
llm = LLaMa(PATH_TO_MODEL, n_layers=50)
```
The concrete VRAM usage per layer depends on the model. If you are low on VRAM start with a lower number like e.g. 10. The library will estimate the total required VRAM after the first attempt.

The final layer may produce a much larger overhead than all previous ones.

## OpenAI

The chat parameter has no influence here.
```python
from GLM import OpenAI, ConversationRoles as cr
# The key can also be set with a shell variable
# you can also use gpt-3.5-turbo-16k and gpt-4
llm = OpenAI("gpt-3.5-turbo", openai_key=YOUR_KEY)

llm.add_tracker_entry(cr.USER, "Whats up?")
llm.add_tracker_entry(cr.ASSISTANT, "just chillin WBU?")
llm.add_tracker_entry(cr.USER, "Utilizing some python stuff for abstracting large language models in an end user centered way even with propietary models")

for i in llm.create_completion_generator(max_tokens=250):
    print(i[0], end="")
```

## Common

### Prompt completion info
Use 
```python
llm.finish_meta
```
to get info about used tokens, time to finish, cost etc.

### Sequence preservation
E.g. with this you can ensure that all Latex equations will only be returned as whole chunks.
```python
generator = llm.create_completion_generator("Write some latex equations", chat=True,preserved_sequences=[{"start":"$$","end":"$$"}])
```


### Function calling

Use the two provided functions to either automatically parse and add functions or manually insert call signatures.

### Saving and loading chats

```python
#save
llm.save_history(PATH)
#load
llm.load_history(PATH)
#reset conversation tracker
llm.reset_tracker()
```

# Installation
```bash
pip3 install git+https://github.com/finnschwall/GLM
```
Requires a working C compiler. For troubleshooting [this](https://github.com/abetlen/llama-cpp-python) is likely the adress.

## Add GPU Acceleration


### Standard
Install Cuda. Download a fitting precompiled binary from 
[here](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/tag/wheels).
When supplying the `n_layers` parameter your GPU should automatically be utilized

### Advanced
*Strongly recommend experience with building*

You need CUDA and cpp build tools.

Build original [library](https://github.com/ggerganov/llama.cpp). It's not strictly necessary.
But gives access to the endless scripts and other stuff.
Also the only way to train LoRA from quantized model is from this fork https://github.com/xaedes/llama.cpp/tree/finetune-lora 
(as of now)

Also makes debugging the next step easier should it fail

Follow [this](https://github.com/abetlen/llama-cpp-python)

When finished supplying the `n_layers` parameter should now utilize your GPU.

# Tracker specification

See [here](https://github.com/finnschwall/GLM/format_specifications.md)