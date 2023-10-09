import os
import numpy as np
import timeit
import ctypes
import warnings
import tqdm
warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)
import psutil
from tqdm.autonotebook import tqdm
from ctypes import POINTER, c_int, c_void_p, cast
from typing import (
    List,
    Optional,
    Union,
    Generator,
    Sequence,
    Iterator,
    Deque,
    Tuple,
    Callable,
    Any
)
import warnings
import numpy as np
import numpy.typing as npt
import pickle
from functools import partial
from .glm import GLM
from .resources import *
#embedding support

#training loras?


    
def _ban_eos_logits_processor(eos_token, input_ids, logits):
    logits[eos_token] = -float('inf')
    return logits

_max_level = 5
_load, _primary_load = True, True
progress_bar = None
_exp_max_char = 700


_meta_dic = {"n_ctx":"Unknown", "n_layer" : "Unknown", "model type":"Unknown", "model size": "Unknown", "model ftype": "Unknown", "general.name": "Unknown"}


n_gpu_layers = -1

_counter = 0
def _log_callback(level: int, text: str, user_data: ctypes.c_void_p):
    global _max_level, _load, _exp_max_char, _primary_load, _counter, _meta_dic
    _counter +=1
    if level < _max_level:
        print(text.decode('utf-8'), end="")
    if _load:
        start = "llm_load_print_meta: "
        start_ggml = "llama_model_load_internal: "
        text = text.decode('utf-8')
        for i in _meta_dic.keys():
            if text.startswith(start+i) or text.startswith(start_ggml+i):
                _meta_dic[i] = text.split("= ")[1][:-1]
        if _primary_load:
            progress_bar.n = max(int((_counter/_exp_max_char)*51),51)
            progress_bar.refresh()
            
        if text==".":
            if _primary_load:
                progress_bar.n = 51
                _primary_load=False
                progress_bar.set_description("Loading stage 2", refresh=True)
            progress_bar.update(1)
    
llama_log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_user_data = ctypes.c_void_p()
_log_callback_pointer = llama_log_callback(_log_callback)



class LLaMa(GLM):
    
    def __init__(self, model_path, n_ctx=2048, verbose=0, n_threads=-1, n_gpu_layers=-1, quantize_format = "auto",is_70b=False, disable_log_hook=False,
                 disable_model_load= False, **kwargs):
        global _max_level, progress_bar, _exp_max_char, _counter, _meta_dic
        super().__init__(model_path, n_ctx=n_ctx, verbose=verbose)
        self.initial_resource_state = get_resource_info()
        _load = True
        _primary_load = True
        _counter = 0

        if disable_model_load:
            self.llm = ""
            return
        
        if quantize_format == "auto":
            if model_path.endswith("gguf"):
                quantize_format = "gguf"
            elif model_path.endswith("bin"):
                quantize_format = "ggml"
            else:
                raise Exception("Could not determine model format. Specify either 'gguf' or 'ggml' for quantize_format.")
        if quantize_format == "ggml":
            try:
                import llama_cpp_ggml
                from llama_cpp_ggml import Llama
                llama_cpp = llama_cpp_ggml
            except:
                from llama_cpp_cuda import Llama
                import llama_cpp_cuda
                llama_cpp = llama_cpp_cuda
        else:
            try:
                import llama_cpp
                from llama_cpp import Llama
                llama_cpp = llama_cpp
            except:
                from llama_cpp_cuda import Llama
                import llama_cpp_cuda
                llama_cpp = llama_cpp_cuda
        if quantize_format=="ggml":
            _exp_max_char = 40
        progress_bar = tqdm(total=165, desc="Loading stage 1", leave=False)

        _max_level=verbose
        globals()["llama_cpp"] = llama_cpp
        if not disable_log_hook:
            llama_cpp.llama_log_set(_log_callback_pointer, _user_data)
        _build_llama(Llama)
        if n_threads == -1:
            n_threads=psutil.cpu_count(logical=False)
        self.n_ctx = n_ctx
        if is_70b:
            self.llm = LlamaBase(model_path=model_path, verbose=verbose, n_ctx=n_ctx, n_threads=n_threads,n_gpu_layers=n_gpu_layers,  seed=-1, n_gqa=8,**kwargs)
        else:
            self.llm = LlamaBase(model_path=model_path, verbose=verbose, n_ctx=n_ctx, n_threads=n_threads,n_gpu_layers=n_gpu_layers,  seed=-1,**kwargs)
        self.ctx = self.llm.ctx


        llama_cpp.llama_print_timings(self.ctx)
        tim = llama_cpp.llama_get_timings(self.ctx)

        self.finish_meta = self.llm.finish_meta
        self.prompt = ""
        self.generated_text = ""
        progress_bar.set_description("Testing inference", refresh=True)
        model_load_time = tim.t_load_ms/1000
        
        self.disable_eos_lproc=llama_cpp.LogitsProcessorList([partial(_ban_eos_logits_processor, self.llm.token_eos())])
        token_generator = self.llm("Once upon a time", max_tokens=15, logits_processor=self.disable_eos_lproc, stream=True)
        
        for i in token_generator:
            progress_bar.update(1)
        progress_bar.n = 165
        progress_bar.set_description("Finished!", refresh=True)
        self.after_load_resource_info = get_resource_info()

        self.load_resource_info = get_resource_diff(self.initial_resource_state, self.after_load_resource_info)
        self.load_resource_info["model_load_time"] = model_load_time
        self.prompt_text_is_str = True

        if _meta_dic["n_layer"]!="Unknown":
            n_layers =  int(_meta_dic["n_layer"])
            n_gpu_layers = min(n_layers, n_gpu_layers)

        if verbose:
            info_str = f"LLM load time:\t{self.load_resource_info['model_load_time']:<5.2f}s\n"\
            f"VRAM used:\t{self.load_resource_info['vram_diff']:<5.0f}mb\n"\
            f"RAM used:\t{self.load_resource_info['ram_diff']:<5.0f}mb\n"\
            f"CPU time:\t{self.load_resource_info['cpu_time_diff']:<5.2f}s"
            if _meta_dic["n_layer"]!="Unknown":
                info_str+=f"\nN layers:\t{n_layers:<5.0f}"
                if n_gpu_layers>0:
                    info_str+=f"\n~VRAM per layer:{self.load_resource_info['vram_diff']/n_gpu_layers:<5.0f}mb"
                    if n_gpu_layers<n_layers:
                        info_str+=f"\nExp max VRAM:\t{self.load_resource_info['vram_diff']/n_gpu_layers*n_layers:<5.0f}mb"
            if  _meta_dic["n_ctx"]!="Unknown":
                info_str+=f"\nMax ctx:\t{_meta_dic['n_ctx']:<5}"    
            if  _meta_dic["model size"]!="Unknown":
                info_str+=f"\nParam count:\t{_meta_dic['model size']:<5}" 
            info_str+=f"\nEstimated t/s:\t{self.finish_meta['t_per_s']['t_gen_per_s']:<5.2f}" 
            
            
            print(info_str)
        self.model_meta_info = _meta_dic
    
    def __del__(self):
        del self.llm


    def get_n_tokens(self,text):
        return len(self.llm.tokenize(bytes(text, "utf-8")))

    def tokenize_as_str(self,text):
        toks_raw = self.llm.tokenize(bytes(text, "utf-8"))
        toks = []
        for i in toks_raw:
            toks.append(self.llm.detokenize([i]).decode("utf-8"))
        return toks

    def tokenize(self,text):
        return self.llm.tokenize(bytes(text, "utf-8"))


    def create_native_generator(self, text, max_tokens=512, stream=True, endless=False, **kwargs):
        #for LLaMA=max tokens -1: shift context, -2 stop when full
        if endless:
            token_generator = self.llm(text, max_tokens=max_tokens, stream=stream,logits_processor=self.disable_eos_lproc, **kwargs)
        else:
            token_generator = self.llm(text, max_tokens=max_tokens, stream=stream, **kwargs)

        return token_generator

    def build_prompt(self):
        prompt = ""
        for i in self.conv_history:
            role = self.symbols[str(i["role"])]
            prompt += f"{role}: {i['content']}\n"

        if "GENERATION_PREFIX" in self.settings and self.settings["GENERATION_PREFIX"] != "":
            prompt += self.settings["GENERATION_PREFIX"]
        prompt = self.replace_symbols(prompt)
        return prompt
    

    # TODO measure time difference for state and context saving
    def save_state_to_disk(self, filename):
        state = self.llm.save_state()
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
            
    def load_state_from_disk(self, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        self.llm.load_state(state)

    def save_ctx_to_disk(self, prompt, path):
        tokens =  self.llm.tokenize(bytes(prompt, "utf-8"))
        llama_context_p = c_void_p
        llama_token = c_int
        llama_token_p = POINTER(llama_token)
        CArray = llama_token * len(tokens)
        c_tokens = CArray(*tokens)
        c_tokens_p = cast(c_tokens, llama_token_p)
        c_ctx = llama_context_p(ctx)
        #LLAMA_API bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count);
        return llama_cpp.llama_save_session_file(c_ctx,  bytes(path,"utf-8"), c_tokens_p, len(tokens))

    def restore_ctx_from_disk(self, path):
        n_token_count_out = c_size_t(0)
        tokens_out = (llama_token * self.n_ctx)()
        c_ctx = llama_context_p(ctx)
        # LLAMA_API bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out)
        return llama_cpp.llama_load_session_file(c_ctx, bytes(path,"utf-8"), tokens_out, self.n_ctx, byref(n_token_count_out))


def _build_llama(_Llama):
    
    from llama_cpp import StoppingCriteria, StoppingCriteriaList, LogitsProcessorList, LogitsProcessor, Completion, CompletionChunk
    class LlamaBase(_Llama):
        def __init__(self, *args, **kwargs):
            super(LlamaBase, self).__init__(*args, **kwargs)
            self.finish_meta = {}
        def _create_completion(
            self,
            prompt: str,
            suffix: Optional[str] = None,
            max_tokens: int = 16,
            temperature: float = 0.8,
            top_p: float = 0.95,
            logprobs: Optional[int] = None,
            echo: bool = False,
            stop: Optional[Union[str, List[str]]] = [],
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            repeat_penalty: float = 1.1,
            top_k: int = 40,
            stream: bool = False,
            tfs_z: float = 1.0,
            mirostat_mode: int = 0,
            mirostat_tau: float = 5.0,
            mirostat_eta: float = 0.1,
            model: Optional[str] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            grammar: Any = None,
        ) -> Union[Iterator[Completion], Iterator[CompletionChunk]]:

            #T=0 -> llama_sample_token_greedy(ctx, candidates) 


            completion_tokens: List[int] = []
            # Add blank space to start of prompt to match OG llama tokenizer
            prompt_tokens: List[int] = self.tokenize(prompt.encode("utf-8")) if prompt != "" else [self.token_bos()]
            text: bytes = b""
            returned_tokens: int = 0
            stop = (
                stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else []
            )
            model_name: str = model if model is not None else self.model_path
    
            llama_cpp.llama_reset_timings(self.ctx)
    
            if len(prompt_tokens) >= llama_cpp.llama_n_ctx(self.ctx):
                raise ValueError(
                    f"Requested tokens ({len(prompt_tokens)}) exceed context window of {llama_cpp.llama_n_ctx(self.ctx)}"
                )
    
            if max_tokens <= 0:
                # Unlimited, depending on n_ctx.
                max_tokens = llama_cpp.llama_n_ctx(self.ctx) - len(prompt_tokens)
    
            # Truncate max_tokens if requested tokens would exceed the context window
            max_tokens = (
                max_tokens
                if max_tokens + len(prompt_tokens) < self._n_ctx
                else (self._n_ctx - len(prompt_tokens))
            )
    
            if stop != []:
                stop_sequences = [s.encode("utf-8") for s in stop]
            else:
                stop_sequences = []
    
            if logprobs is not None and self.params.logits_all is False:
                raise ValueError(
                    "logprobs is not supported for models created with logits_all=False"
                )
    
            if self.cache:
                try:
                    cache_item = self.cache[prompt_tokens]
                    cache_prefix_len = Llama.longest_token_prefix(
                        cache_item.input_ids.tolist(), prompt_tokens
                    )
                    eval_prefix_len = Llama.longest_token_prefix(
                        self._input_ids.tolist(), prompt_tokens
                    )
                    if cache_prefix_len > eval_prefix_len:
                        self.load_state(cache_item)
                        if self.verbose:
                            print("Llama._create_completion: cache hit", file=sys.stderr)
                except KeyError:
                    if self.verbose:
                        print("Llama._create_completion: cache miss", file=sys.stderr)
    
            self.finish_meta["finish_reason"] = "length"
            multibyte_fix = 0
            for token in self.generate(
                prompt_tokens,
                top_k=top_k,
                top_p=top_p,
                temp=temperature,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                grammar=grammar,
            ):
                if token == self._token_eos:
                    text = self.detokenize(completion_tokens)
                    self.finish_meta["finish_reason"] = "stop"
                    break
    
                completion_tokens.append(token)
    
                all_text = self.detokenize(completion_tokens)
    
                # Contains multi-byte UTF8
                for k, char in enumerate(all_text[-3:]):
                    k = 3 - k
                    for num, pattern in [(2, 192), (3, 224), (4, 240)]:
                        # Bitwise AND check
                        if num > k and pattern & char == pattern:
                            multibyte_fix = num - k
    
                # Stop incomplete bytes from passing
                if multibyte_fix > 0:
                    multibyte_fix -= 1
                    continue
    
                any_stop = [s for s in stop_sequences if s in all_text]
                if len(any_stop) > 0:
                    first_stop = any_stop[0]
                    text = all_text[: all_text.index(first_stop)]
                    self.finish_reason = "stop"
                    break
    
                if stream:
                    remaining_tokens = completion_tokens[returned_tokens:]
                    remaining_text = self.detokenize(remaining_tokens)
                    remaining_length = len(remaining_text)
    
                    # We want to avoid yielding any characters from
                    # the generated text if they are part of a stop
                    # sequence.
                    first_stop_position = 0
                    for s in stop_sequences:
                        for i in range(min(len(s), remaining_length), 0, -1):
                            if remaining_text.endswith(s[:i]):
                                if i > first_stop_position:
                                    first_stop_position = i
                                break
    
                    token_end_position = 0
    
                    if logprobs is not None:
                        # not sure how to handle this branch when dealing
                        # with CJK output, so keep it unchanged
                        for token in remaining_tokens:
                            token_end_position += len(self.detokenize([token]))
                            # Check if stop sequence is in the token
                            if token_end_position > (remaining_length - first_stop_position):
                                break
                            token_str = self.detokenize([token]).decode(
                                "utf-8", errors="ignore"
                            )
                            text_offset = len(prompt) + len(
                                self.detokenize(completion_tokens[:returned_tokens])
                            )
                            token_offset = len(prompt_tokens) + returned_tokens
                            logits = self._scores[token_offset - 1, :].tolist()
                            current_logprobs = Llama.logits_to_logprobs(logits)
                            sorted_logprobs = list(
                                sorted(
                                    zip(current_logprobs, range(len(current_logprobs))),
                                    reverse=True,
                                )
                            )
                            top_logprob = {
                                self.detokenize([i]).decode(
                                    "utf-8", errors="ignore"
                                ): logprob
                                for logprob, i in sorted_logprobs[:logprobs]
                            }
                            top_logprob.update({token_str: current_logprobs[int(token)]})
                            logprobs_or_none = {
                                "tokens": [
                                    self.detokenize([token]).decode(
                                        "utf-8", errors="ignore"
                                    )
                                ],
                                "text_offset": [text_offset],
                                "token_logprobs": [current_logprobs[int(token)]],
                                "top_logprobs": [top_logprob],
                            }
                            returned_tokens += 1
                            yield  self.detokenize([token]).decode(
                                            "utf-8", errors="ignore")
                    else:
                        while len(remaining_tokens) > 0:
                            decode_success = False
                            for i in range(1, len(remaining_tokens) + 1):
                                try:
                                    bs = self.detokenize(remaining_tokens[:i])
                                    ts = bs.decode('utf-8')
                                    decode_success = True
                                    break
                                except UnicodeError:
                                    pass
                            else:
                                break
                            if not decode_success:
                                # all remaining tokens cannot be decoded to a UTF-8 character
                                break
                            token_end_position += len(bs)
                            if token_end_position > (remaining_length - first_stop_position):
                                break
                            remaining_tokens = remaining_tokens[i:]
                            returned_tokens += i
                            yield ts
                            
    
                if len(completion_tokens) >= max_tokens:
                    text = self.detokenize(completion_tokens)
                    self.finish_meta["finish_reason"] = "length"
                    break
    
            if stopping_criteria is not None and stopping_criteria(
                self._input_ids, self._scores[-1, :]
            ):
                text = self.detokenize(completion_tokens)
                self.finish_meta["finish_reason"] ="stop"

            tim = llama_cpp.llama_get_timings(self.ctx)
            tot_ms = tim.t_eval_ms+ tim.t_p_eval_ms+tim.t_sample_ms+tim.t_load_ms
            try:
                t_intake_per_s=tim.n_p_eval/tim.t_p_eval_ms*1000
            except:
                t_intake_per_s = 0
            try:
                t_total_per_s = (tim.n_p_eval+tim.n_eval+1)/tot_ms*1000
            except:
                t_total_per_s = 0
            try:
                t_gen_per_s = (tim.n_eval)/tim.t_eval_ms*1000
            except:
                t_gen_per_s = 0
            timing_str = f"\nTotal speed:  tokens:\t{tot_ms:<8.0f}ms / {tim.n_p_eval+tim.n_eval+1:<5.0f} t = {t_total_per_s:<7.2f} t/s\n"\
    f"Prompt intake speed\t{tim.t_p_eval_ms:<8.0f}ms / {tim.n_p_eval:<5} t = {t_intake_per_s:<7.2f} t/s\n"\
    f"Generation speed:\t{tim.t_eval_ms:<8.0f}ms / {tim.n_eval:<5} t = {t_gen_per_s:<7.2f} t/s"

            
            self.finish_meta["tokens"] = {"Prompt tokens": tim.n_p_eval, "Generated tokens": tim.n_eval}
            self.finish_meta["timings"] = {"ms for prompt" :tim.t_p_eval_ms, "ms for generation": tim.t_eval_ms, "total time": tot_ms}
            self.finish_meta["t_per_s"] = {"t_intake_per_s":t_intake_per_s,"t_total_per_s":t_total_per_s, "t_gen_per_s":t_gen_per_s}
            self.finish_meta["timing_str"] = timing_str
            
            
            if stream:
                remaining_tokens = completion_tokens[returned_tokens:]
                all_text = self.detokenize(remaining_tokens)
                any_stop = [s for s in stop_sequences if s in all_text]
                if len(any_stop) > 0:
                    end = min(all_text.index(stop) for stop in any_stop)
                else:
                    end = len(all_text)
    
                token_end_position = 0
                for token in remaining_tokens:
                    token_end_position += len(self.detokenize([token]))
    
                    logprobs_or_none: Optional[CompletionLogprobs] = None
                    if logprobs is not None:
                        token_str = self.detokenize([token]).decode(
                            "utf-8", errors="ignore"
                        )
                        text_offset = len(prompt) + len(
                            self.detokenize(completion_tokens[:returned_tokens])
                        )
                        token_offset = len(prompt_tokens) + returned_tokens - 1
                        logits = self._scores[token_offset, :].tolist()
                        current_logprobs = Llama.logits_to_logprobs(logits)
                        sorted_logprobs = list(
                            sorted(
                                zip(current_logprobs, range(len(current_logprobs))),
                                reverse=True,
                            )
                        )
                        top_logprob = {
                            self.detokenize([i]).decode("utf-8", errors="ignore"): logprob
                            for logprob, i in sorted_logprobs[:logprobs]
                        }
                        top_logprob.update({token_str: current_logprobs[int(token)]})
                        logprobs_or_none = {
                            "tokens": [
                                self.detokenize([token]).decode("utf-8", errors="ignore")
                            ],
                            "text_offset": [text_offset],
                            "token_logprobs": [current_logprobs[int(token)]],
                            "top_logprobs": [top_logprob],
                        }
    
                    if token_end_position >= end:
                        last_text = self.detokenize([token])
                        if token_end_position == end - 1:
                            break
                        returned_tokens += 1
                        yield last_text[: len(last_text) - (token_end_position - end)].decode("utf-8", errors="ignore")#, logprobs_or_none
                        yield ""
                        break
                    returned_tokens += 1
                    yield self.detokenize([token]).decode(
                                    "utf-8", errors="ignore"
                                ), logprobs_or_none 
                    yield ""
                if self.cache:
                    if self.verbose:
                        print("Llama._create_completion: cache save", file=sys.stderr)
                    self.cache[prompt_tokens + completion_tokens] = self.save_state()
                    print("Llama._create_completion: cache saved", file=sys.stderr)
                return
    
            if self.cache:
                if self.verbose:
                    print("Llama._create_completion: cache save", file=sys.stderr)
                self.cache[prompt_tokens + completion_tokens] = self.save_state()
    
            text_str = text.decode("utf-8", errors="ignore")
    
            if echo:
                text_str = prompt + text_str
    
            if suffix is not None:
                text_str = text_str + suffix
    
            logprobs_or_none: Optional[CompletionLogprobs] = None
            if logprobs is not None:
                text_offset = 0 if echo else len(prompt)
                token_offset = 0 if echo else len(prompt_tokens[1:])
                text_offsets: List[int] = []
                token_logprobs: List[Optional[float]] = []
                tokens: List[str] = []
                top_logprobs: List[Optional[Dict[str, float]]] = []
    
                if echo:
                    # Remove leading BOS token
                    all_tokens = prompt_tokens[1:] + completion_tokens
                else:
                    all_tokens = completion_tokens
    
                all_token_strs = [
                    self.detokenize([token]).decode("utf-8", errors="ignore")
                    for token in all_tokens
                ]
                all_logprobs = [
                    Llama.logits_to_logprobs(row.tolist()) for row in self._scores
                ][token_offset:]
                for token, token_str, logprobs_token in zip(
                    all_tokens, all_token_strs, all_logprobs
                ):
                    text_offsets.append(text_offset)
                    text_offset += len(token_str)
                    tokens.append(token_str)
                    sorted_logprobs = list(
                        sorted(
                            zip(logprobs_token, range(len(logprobs_token))), reverse=True
                        )
                    )
                    token_logprobs.append(logprobs_token[int(token)])
                    top_logprob: Optional[Dict[str, float]] = {
                        self.detokenize([i]).decode("utf-8", errors="ignore"): logprob
                        for logprob, i in sorted_logprobs[:logprobs]
                    }
                    top_logprob.update({token_str: logprobs_token[int(token)]})
                    top_logprobs.append(top_logprob)
                # Weird idosincracy of the OpenAI API where
                # token_logprobs and top_logprobs are null for
                # the first token.
                if echo and len(all_tokens) > 0:
                    token_logprobs[0] = None
                    top_logprobs[0] = None
                logprobs_or_none = {
                    "tokens": tokens,
                    "text_offset": text_offsets,
                    "token_logprobs": token_logprobs,
                    "top_logprobs": top_logprobs,
                }
            yield text_str

    globals()["LlamaBase"] = LlamaBase


    
