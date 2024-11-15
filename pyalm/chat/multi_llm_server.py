import json

from pyalm import ConversationTracker
from rixaplugin import variables as var
from rixaplugin.decorators import global_init, worker_init, plugfunc
from rixaplugin import worker_context, execute, async_execute

from rixaplugin.internal.memory import _memory
from rixaplugin.internal import api as internal_api
import rixaplugin
from . import system_msg_templates
import logging
import rixaplugin.sync_api as api
import os
# openai_key = var.PluginVariable("OPENAI_KEY", str, readable=var.Scope.LOCAL)

llm_logger = logging.getLogger("rixa.llm_server")
import time

@worker_init()
def worker_init():
    try:
        import multiprocessing
        try:
            proc_id = int(multiprocessing.current_process().name.replace("ForkProcess-",""))
        except:
            proc_id=1
        with open("models.json","r") as f:
            models = json.load(f)
        worker_context.proc_id = proc_id
        path = models[proc_id-1].pop("path")
        if "quantize_kv" in models[proc_id-1]:
            del models[proc_id-1]["quantize_kv"]
            import llama_cpp
            add_kwargs = {"type_k" : llama_cpp.GGML_TYPE_Q4_0,
            "type_v" : llama_cpp.GGML_TYPE_Q4_0,
            "flash_attn" : True}
            models[proc_id-1].update(add_kwargs)
        if "gpus" in models[proc_id-1]:
            gpus = models[proc_id-1].pop("gpus")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
        from pyalm.models.llama import LLaMa
        llm = LLaMa(path, **models[proc_id-1])
        worker_context.llm = llm
    except Exception as e:
        print(repr(e))
        raise e


# @plugfunc()
# def test_cmd():


@plugfunc()
def get_total_tokens():
    return worker_context.llm.total_tokens

@plugfunc()
def create_completion_plugin(*args, **kwargs):
    # return "worked"
    start_time = time.time()
    print(worker_context.proc_id, " started generation")
    response, metadata = worker_context.llm.create_completion_plugin(*args, **kwargs)
    total_time = time.time() - start_time
    print(worker_context.proc_id, " finished generation after ", round(total_time,3), " s")
    return response, metadata

@plugfunc()
def translate_last_message(conversation_history, to_english=True):
    translator_message = system_msg_templates.translation_to_EN if to_english else system_msg_templates.translation_to_DE
    tracker = ConversationTracker.from_yaml(conversation_history)
    worker_context.llm.conversation_history = tracker
    llm_response = worker_context.llm.create_completion(translator_message, temperature=0)
    return llm_response


@plugfunc()
def get_preprocessing_json(conversation_history, knowledge_retrieval_domain, system_msg=None):
    user_api = internal_api.get_api()
    preprocessor_msg = system_msg_templates.preprocessor_msg
    preprocessor_msg = preprocessor_msg.replace("[[document_tags]]", f"{knowledge_retrieval_domain}")
    preprocessor_msg = preprocessor_msg.replace("[[functions]]",
                                                f"{_memory.get_functions_as_str(user_api.scope, short=False, include_docstr=False)}")
    # preprocessor_msg = preprocessor_msg.replace("[[system_msg]]", f"{system_msg}")
    worker_context.llm.user_symbols["USR_SYSTEM_MSG"] = "" if not system_msg else system_msg

    tracker = ConversationTracker.from_yaml(conversation_history)
    worker_context.llm.conversation_history = tracker
    llm_response = worker_context.llm.create_completion(preprocessor_msg,temperature=0)
    try:
        preprocessor_json = json.loads(llm_response)
    except:
        llm_logger.exception(f"Could not parse preprocessor response")
        # api.display_in_chat(text="An unrecoverable error has occurred. You can try again after reloading the page", role="partial")
        return None
    api.display_in_chat(text="Preprocessing done. Starting response generation...", role="partial")
    return preprocessor_json, worker_context.llm.finish_meta
