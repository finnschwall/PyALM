<img src="https://cdn-icons-png.flaticon.com/512/10721/10721994.png" alt="drawing" width="75"/>
<b>You are entering an active construction zone!</b>

# FINAL COMMIT
After this commit history will be purged to allow for a restructure of the repository, to be more appropriate for a public release!


<p align="center">
<img src="icon.png" alt="drawing" width="250"/>
</p>

# Overview
PyALM: Python Abstract Language Model is a library that provides a unified interface for running large
language models (LLMs). It does not aim to replace libraries like e.g. langchain but rather
provide some distinctive enhancements for programming a chat deployment. Among them are
* Text streaming :white_check_mark:
* Sequence preservation: with PyALM you can e.g. force equations inside dollar signs only to be streamed as chunk :white_check_mark:
* Function calling: Allow the model to call single functions or chained calls with variables using a python like syntax. :white_check_mark:
* Unified advanced chat history. There is an [extensive format](https://github.com/finnschwall/PyALM/blob/main/format_specifications.md)
baked into the abstraction layer that allows for infos on the user, user interactions, feedback, context, service integration and much more :ballot_box_with_check: (done mostly)
* Citation ability: PyALM can provide models with facts and context from various sources and reverse search for the user <p style="color:orange">WIP</p>
* Sentiment analysis for truly user based experiences <p style="color:red">Not started yet</p>

PyALM is part of the broader development of the [RIXA framework](https://github.com/finnschwall/RIXA). It
provides RIXAs natural language abilities.
Have a look if you want to see a cool use case of this project or LLMs in general.
# Usage
This is a short best of. More infos and examples are in the [docs](https://finnschwall.github.io/PyALM/).

## LLaMa

If you don't have one: Download a model. Good address is e.g. [TheBloke](https://huggingface.co/TheBloke).
On CPU only I recommend 7B models. 13B only if you are really patient.

### Chat
```python
from pyalm import LLaMa, ConversationRoles as cr
llm = LLaMa(PATH_TO_MODEL)

llm.add_tracker_entry(cr.USER, "Whats up?")
llm.add_tracker_entry(cr.ASSISTANT, "just chillin WBU?")
# This is equivalent to adding a user tracker entry and not specifying a text
for i in llm.create_generator("Utilizing some python stuff for abstracting large language models in an end user centered way"):
    print(i[0], end="")
```

### Raw text

With LLaMa and AlephALpha raw input into the model is also possible. You need to specify `chat=False` for this.
```python
from pyalm import LLaMa
llm = LLaMa(PATH_TO_MODEL)

for i in llm.create_generator("O Captain! my Captain! our fearful trip", max_tokens=250, chat=False):
    print(i[0], end="")
```



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

for i in llm.create_generator(max_tokens=250):
    print(i[0], end="")
```

## Common

### Completion only
```python
completion = llm.create_completion()
```
Basically equivalent in final result. Functions will still be called automatically if not disabled.

### Prompt completion info
Use 
```python
llm.finish_meta
```
to get info about used tokens, time to finish, cost etc.

### Sequence preservation
E.g. with this you can ensure that all Latex equations will only be returned as whole chunks.
```python
generator = llm.create_generator("Write some latex equations", chat=True,preserved_sequences=[{"start":"$$","end":"$$"}])
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

## Add GPU Acceleration for local models
See [here](https://finnschwall.github.io/PyALM/modules/module_llama.html#installing-hardware-acceleration)

# Running large models without great hardware
See [here](https://finnschwall.github.io/PyALM/modules/module_llama.html#how-to-use-without-gpu)

# Tracker specification
See [here](https://github.com/finnschwall/PyALM/blob/main/format_specifications.md)


# How to build the docs
```bash
pip3 install sphinx myst-parser sphinxcontrib-mermaid
#cd to docs
make html
```

