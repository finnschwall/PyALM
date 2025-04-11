import json
import random

from pyalm import ConversationTracker, ConversationRoles
from rixaplugin import variables as var
from rixaplugin.decorators import global_init, worker_init, plugfunc
from rixaplugin import worker_context, execute, async_execute
from rixaplugin.internal.memory import _memory
from rixaplugin.internal import api as internal_api
import rixaplugin
from . import system_msg_templates
import logging
import rixaplugin.sync_api as api
import time

llm_logger = logging.getLogger("rixa.llm_server")
import time

@worker_init()
def worker_init():
    import multiprocessing
    try:
        proc_id = int(multiprocessing.current_process().name.replace("ForkProcess-",""))
    except:
        proc_id=1
    worker_context.proc_id = proc_id


@plugfunc()
def get_total_tokens():
    return -1


@plugfunc()
def create_completion_plugin(*args, **kwargs):
    wait_time = random.randint(5, 30)
    print(f"Fake completion started {worker_context.proc_id}. Waiting {wait_time} seconds.")

    time.sleep(wait_time)
    tracker = kwargs["conv_tracker"]
    tracker.metadata = {"model_name" : "fake_llm_server"}
    # tracker = ConversationTracker.from_yaml(tracker)
    response = "This is a fake response"
    tracker.add_entry(response, ConversationRoles.ASSISTANT)
    metadata = {"total_tokens": 0, "total_time": wait_time}
    print(f"Fake completion finished {worker_context.proc_id}")
    return tracker, metadata

@plugfunc()
def translate_last_message(conversation_history, to_english=True):
    return None


@plugfunc()
def get_preprocessing_json(conversation_history, knowledge_retrieval_domain, system_msg=None):
    return None

_memory.rename_plugin("fake_llm_server", "openai")
