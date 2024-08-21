import json

from pyalm import ConversationTracker
from rixaplugin import variables as var
from rixaplugin.decorators import global_init, worker_init, plugfunc
from rixaplugin import worker_context, execute, async_execute
from pyalm.models.openai import OpenAI
from rixaplugin.internal.memory import _memory
from rixaplugin.internal import api as internal_api
from . import system_msg_templates
import logging
import rixaplugin.sync_api as api
openai_key = var.PluginVariable("OPENAI_KEY", str, readable=var.Scope.LOCAL)
llm_logger = logging.getLogger("rixa.llm_server")
@worker_init()
def worker_init():
    #gpt-4-turbo
    #gpt-4-32k-0613

    #gpt-4o-2024-05-13
    #gpt-4-32k-0613
    llm = OpenAI("gpt-4o-2024-05-13", openai_key.get())
    worker_context.llm = llm


@plugfunc()
def get_total_tokens():
    return worker_context.llm.total_tokens

@plugfunc()
def create_completion_plugin(*args, **kwargs):
    response, metadata =worker_context.llm.create_completion_plugin(*args, **kwargs)
    # print(worker_context.llm.finish_meta)
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
    return preprocessor_json
