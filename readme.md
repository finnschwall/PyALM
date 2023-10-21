<img src="https://cdn-icons-png.flaticon.com/512/6261/6261561.png" alt="drawing" width="75"/>
<b>You are entering an active construction zone!</b>

The project is in an alpha stadium

<p align="center">
<img src="icon.png" alt="drawing" width="250"/>
</p>

# Overview
PyALM: Python Abstract Language Model is a library that provides a unified interface for running large
language models (LLMs). It does not aim to replace libraries like e.g. langchain but rather
provide some distinctive enhancements for programming a chat deployment. Among them are
* Text streaming
* Sequence preservation: with PyALM you can force equations inside dollar signs only to be streamed as chunk
* Function calling: Allow the model to call single functions or chained calls with variables using a python like syntax.
* Unified advanced chat history. There is an [extensive format](https://github.com/finnschwall/PyALM/blob/main/format_specifications.md)
baked into the abstraction layer that allows for infos on the user, user interactions, feedback, context, service integration and much more
* Citation ability: PyALM can provide models with facts and context from various sources and reverse search for the user
* Sentiment analysis for truly user based experiences

PyALM is part of the broader development of the [RIXA framework](https://github.com/finnschwall/RIXA). It
provides RIXAs natural language abilities.
Have a look if you want to see a cool use case of this project or LLMs in general.
# Usage
This is a short best of. More infos and examples are in the [docs](https://finnschwall.github.io/PyALM/).

## LLaMa

If you don't have one: Download a model. Good address is e.g. [TheBloke](https://huggingface.co/TheBloke).
On CPU only I recommend 7B models. 13B only if you are really patient.

In the [C++ library](https://github.com/ggerganov/llama.cpp) there are various scripts that can be used to obtain more models e.g. by quantizing models. Although that does require considerable resources.
### Chat
```python
from pyalm import LLaMa, ConversationRoles as cr
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
from pyalm import LLaMa
llm = LLaMa(PATH_TO_MODEL)

for i in llm.create_completion_generator("O Captain! my Captain! our fearful trip", max_tokens=250):
    print(i[0], end="")
```

### Hardware acceleration

For utilizing the GPU install GPU support and use the `n_layers` parameter.

```python
from pyalm import LLaMa
llm = LLaMa(PATH_TO_MODEL, n_layers=50)
```
The concrete VRAM usage per layer depends on the model. If you are low on VRAM start with a lower number like e.g. 10. The library will estimate the total required VRAM after the first attempt.

The final layer may produce a much larger overhead than all previous ones.

### 70b
```python
from pyalm import LLaMa
llm = LLaMa(PATH_TO_MODEL, is_70b=True)
```
Will lead to errors for non 70B models. Without proper GPU this is a futile endeavor.
## OpenAI

The chat parameter has no influence here.
```python
from pyalm import OpenAI, ConversationRoles as cr
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
pip3 install git+https://github.com/finnschwall/PyALM#egg=pyalm[normal]
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

See [here](https://github.com/finnschwall/PyALM/blob/main/format_specifications.md)

# Running larger local models without GPU
Due to the nature of the task you will come only this far with CPU-only. You can use a backend like exllama
that has more aggressive optimizations, use lower bit quantizations and so on.

Be aware though: A lot of the more effective optimizations cause quality degradation in various degrees.

## So what then?

### Just inference
If you don't want to code but just infer you could use third party providers like
e.g. Aleph-Alpha. As they usually offer their own playground the usefulness of this framework is quite limited.

### Coding+Inference
* Google colab is a good start. GPU availability may be limited. Also you can only have one notebook so larger
projects are difficult.
* Kaggle offers free GPU accelerated notebooks
* There is a lot more
## Secret dev tip
[Saturncloud](https://saturncloud.io/)

A lot of this and other RIXA stuff was developed there. Incredibly helpful for background tasks.
You get 150 free compute hours/month.
There are no problems with GPU availability. But most importantly it allows for full project structures and
temporary deployments into the web.

CUDA is preinstalled (11.7) so you can use the preinstalled binaries with an identifier like this
`cu117-cp39-cp39-linux_x86_64`

The free version 'only' contains 16 GB VRAM + 16 GB RAM so ~6B quantized 30B models is the absolute maximum
you can get out.

# How to build the docs
```bash
pip3 install sphinx myst-parser sphinxcontrib-mermaid
#cd to docs
make html
```

