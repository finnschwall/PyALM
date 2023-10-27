# Core
## Introduction
Core components. Most notably ALM which is the Class from which all models derive.
All functions that are common amongst models are here. 

## ALM DIY
Override
`build_prompt` and at least one of
`create_native_generator create_native_completion`. Implementing `create_native_generator` but not
`create_native_completion` will not lead to automatic availability of the latter.

Ideally you also override
`detokenize tokenize tokenize_as_str get_n_tokens`

The easiest example of all this, is the OpenAI class.

## Documentation
```{eval-rst}  
.. automodule:: pyalm.alm
   :members:
   :undoc-members:
```


