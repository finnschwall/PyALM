# Aleph Alpha
## Introduction

Use with
```python
from pyalm import AlephAlpha
llm = AlephAlpha("luminous-extended-control", aleph_alpha_key=KEY)
```
Alternatively the key can be ignored and set via the env var `AA_TOKEN`.
You can set the used model to a non supported one or change it anytime
via `llm.model = NAME`
Cost can be accessed via `llm.finish_meta` after a call or with
```python
AlephAlpha.pricing
AlephAlpha.pricing_factors
AlephAlpha.pricing_meta
AlephAlpha.pricing_img
```


## Documentation
```{eval-rst}  
.. automodule:: pyalm.models.alephalpha
   :members:
   :undoc-members:
```


