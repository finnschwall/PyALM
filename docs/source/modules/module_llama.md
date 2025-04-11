# Local LlaMa models

## Background and resources
This is built using [llama.cpp](https://github.com/ggerganov/llama.cpp) and it's python bindings
from [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

Documentation is the llama [header](https://github.com/ggerganov/llama.cpp/blob/master/llama.h).



## Acquiring models
You need a quantized model. For raw pytorch models use the huggingface ALM (not finished).

### Where to look
Good address is e.g. [TheBloke](https://huggingface.co/TheBloke).

### Quantizing a model
Look in [C library](https://github.com/ggerganov/llama.cpp). Quantization is resource hungry. Can be used
to make any Llama based model usable and generally at quite the significant speed increase.

## Usage info
Basic
```python
from pyalm import LLaMa
llm = LLaMa(PATH_TO_QUANTIZED_MODEL_FILE)
```
Everything else is mostly model dependent. You can find that out via a model card. Alternatively
you can load the model for a single time. The library will obtain everything there is to find out from the file



### CPU only
CPU support is automatic. Perfomance can be controlled via `n_threads`. If not set the library will take whatever it can get.
Lower quantizations of the same model are faster but quality can suffer immensely.

### GPU only or mixed
`n_gpu_layers` is what controls how much of the model is offloaded to a GPU.
It has no effect on versions that are not compiled with CUBLAS.
The required VRAM per layer is model dependent and can be found out via a first load with a low-ish
value like e.g. 10-20 layers.

The final layer may produce a much larger overhead than all previous ones and is not accounted for in the
total VRAM usage estimate.

### 70b
```python
from pyalm import LLaMa
llm = LLaMa(PATH_TO_MODEL, is_70b=True)
```
Will lead to errors for non 70B models. Without proper GPU this is a futile endeavor.


## Documentation
```{eval-rst}  
.. automodule:: pyalm.models.llama
   :members:
   :undoc-members:
```

## Installing hardware acceleration
CPU always works but is not _goal oriented_ for models > 13B params. There are speed-ups available for
cpu only via providing better BLAS libraries. Look at [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
for more info.

### GPU-Standard
Install Cuda. Download a fitting precompiled wheel from 
[here](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/tag/wheels) and install it.
When supplying the `n_layers` parameter your GPU should automatically be utilized

### GPU-Advanced
*Recommend experience with building*

You need CUDA and cpp build tools.

Build original [library](https://github.com/ggerganov/llama.cpp). It's not strictly necessary.
But gives access to the endless scripts and other stuff.
Also the only way to train LoRA from quantized model is from this fork https://github.com/xaedes/llama.cpp/tree/finetune-lora 
(as of now)

And makes debugging the next step easier should it fail

Follow [this](https://github.com/abetlen/llama-cpp-python)

When finished supplying the `n_layers` parameter should now utilize your GPU.



## How to use without GPU
Due to the nature of the task you will come only this far with CPU-only. You can use a backend like exllama
that has more aggressive optimizations, use lower bit quantizations and so on.

Be aware though: A lot of the more effective optimizations cause quality degradation in various degrees.

### Just inference
If you don't want to code but just infer you could use third party providers like
e.g. Aleph-Alpha. As they usually offer their own playground the usefulness of this framework is quite limited.
But I am glad to be of help anyway.

### Coding+Inference
* Google colab is a good start. GPU availability may be limited. Also you can only have one notebook so larger
projects are difficult.
* Kaggle offers free GPU accelerated notebooks
* There is a lot more
### Not-so-secret dev tip
[Saturncloud](https://saturncloud.io/)

A lot of this and other RIXA stuff was developed there. Incredibly helpful for background tasks.
You get 150 free compute hours/month.
There are no problems with GPU availability. But most importantly it allows for full project structures and
temporary deployments into the web.

CUDA is preinstalled (11.7) so you can use the preinstalled binaries with an identifier like this
`cu117-cp39-cp39-linux_x86_64`

The free version 'only' contains 16 GB VRAM + 16 GB RAM so ~6B quantized 30B models is the absolute maximum
you can get out.