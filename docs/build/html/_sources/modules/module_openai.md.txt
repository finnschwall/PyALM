# OpenAI
## Intro
Use with
```python
from pyalm import OpenAI
llm = OpenAI("gpt4", openai_key=KEY)
```
Alternatively the key can be ignored and set via the env var `OPENAI_API_KEY`. You can set the used model to a non supported one
via `llm.model = NAME`
Cost can be accessed via `llm.finish_meta` after a call or with `OpenAI.pricing` and
`OpenAI.pricing_meta` to get the rates.

Native OpenAI function calls are currently not supported

## Documentation
```{eval-rst}  
.. automodule:: pyalm.models.openai
   :members:
   :undoc-members:
```


