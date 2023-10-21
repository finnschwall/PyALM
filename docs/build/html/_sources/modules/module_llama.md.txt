# Local LlaMa models

## Background and resources
This is built using [llama.cpp](https://github.com/ggerganov/llama.cpp) and it's python bindings
from [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

Documentation is the llama [header](https://github.com/ggerganov/llama.cpp/blob/master/llama.h).

## Usage info
Basic
```python
from pyalm import LLaMa
llm = LLaMa(PATH_TO_QUANTIZED_MODEL_FILE)
```
Everything else is mostly model dependent. You can find that out via a model card. Alternatively
you can load the model for a single time. The library will obtain everything there is to find out from the file

### Quantize a model
Look in [C library](https://github.com/ggerganov/llama.cpp). Quantization is resource hungry.

### CPU
CPU support is automatic. Perfomance can be controlled via `n_threads`. If not set the library will take whatever it can get.
Lower quantizations of the same model are faster but quality can suffer immensely.

Everything above 13B is basically unrunnable. 7B gives decent speed but questionable performance.
### GPU
`n_gpu_layers` is what controls how much of the model is offloaded to a GPU.
It has no effect on versions that are not compiled with CUBLAS.
The required VRAM per layer is model dependent and can be found out via a first load with a low-ish
value like e.g. 10-20 layers. 


## Documentation
```{eval-rst}  
.. automodule:: pyalm.llama
   :members:
   :undoc-members:
```


