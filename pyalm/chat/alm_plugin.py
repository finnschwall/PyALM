import datetime
import json
import re
import requests

from rixaplugin import variables as var
from rixaplugin.data_structures.rixa_exceptions import QueueOverflowException, PluginNotFoundException, \
    RemoteTimeoutException, RemoteOfflineException
from rixaplugin.decorators import global_init, worker_init, plugfunc
from rixaplugin import worker_context, execute, async_execute
import yaml
from rixaplugin.internal.memory import _memory

import rixaplugin.async_api as api

from pyalm.models.openai import OpenAI
from pyalm import ConversationTracker, ConversationRoles

import logging
from rixaplugin.internal import api as internal_api
import os
import aiohttp

llm_logger = logging.getLogger("rixa.llm_server")

# from rixaplugin.examples import knowledge_base

import time

openai_key = var.PluginVariable("OPENAI_KEY", str, readable=var.Scope.LOCAL)
deepl_key = var.PluginVariable("DEEPL_KEY", str, readable=var.Scope.LOCAL)
max_tokens = var.PluginVariable("MAX_TOKENS", int, 4096, readable=var.Scope.LOCAL)
chat_store_loc = var.PluginVariable("chat_store_loc", str, default=None)
multiplexing = var.PluginVariable("multiplexing", bool, default=False, readable=var.Scope.USER, writable=var.Scope.USER,
                                  user_facing_name="Enable Multiplexing")
translation_layer = var.PluginVariable("translation_layer", str, default="None", options=["None", "deepl", "LLM"],
                                       readable=var.Scope.USER, writable=var.Scope.USER)
enable_knowledge_retrieval_var = var.PluginVariable("enable_knowledge_retrieval", bool, default=True,
                                                    readable=var.Scope.USER, writable=var.Scope.USER)
nlp_engine_options_var = var.PluginVariable("NLP_ENGINE_OPTIONS", str, default="openai")
nlp_engine_options= nlp_engine_options_var.get().split(",")
nlp_engine = var.PluginVariable("nlp_engine", str, default=nlp_engine_options[0], readable=var.Scope.USER,
                                writable=var.Scope.USER, options=nlp_engine_options)

async def translate_message(message, context=None, target_lang="DE"):
    url = 'https://api-free.deepl.com/v2/translate'
    data = {
        "auth_key": deepl_key.get(),
        "text": message,
        "target_lang": target_lang
    }
    if context:
        data["context"] = context
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as response:
            response_json = await response.json()
            return response_json["translations"][0]["text"]

async def load_balanced_request(func_name, args=None, kwargs=None):
    current_idx = nlp_engine_options.index(nlp_engine.get())
    did_balancing = False
    did_replace=False
    success = False
    success_backend = ""
    orig_backend = nlp_engine_options[current_idx]
    for idx in range(current_idx, len(nlp_engine_options)):
        try:
            fut = await async_execute(func_name, nlp_engine_options[idx], args=args, kwargs=kwargs, return_future=True)
            ret_val = await fut
            success = True
            success_backend = nlp_engine_options[idx]
            break
        except QueueOverflowException as e:
            did_balancing = True
            continue
        except PluginNotFoundException as e:
            did_replace = True
            continue
        except RemoteTimeoutException as e:
            did_replace = True
            continue
        except RemoteOfflineException as e:
            did_replace = True
            continue
    if not success:
        raise Exception("Server overload")
    else:
        if did_balancing:
            # await api.show_message(f"Msg redirected to {success_backend} for load balancing", "info")
            llm_logger.info(f"Msg redirected to {success_backend} for load balancing")
        if did_replace:
            llm_logger.info(f"Backend '{orig_backend}' not found/available, redirected to {success_backend}")
            # await api.show_message(f"Backend '{orig_backend}' not found/available, redirected to {success_backend}","info")
        return ret_val



@plugfunc()
async def generate_text(conversation_tracker_yaml, enable_function_calling=True, enable_knowledge_retrieval=True,
                        knowledge_retrieval_domain=None, system_msg=None, username=None):
    """
    Generate text based on the conversation tracker and available functions

    :param conversation_tracker_yaml: The conversation tracker in yaml format
    :param available_functions: A list of available functions
    """
    start_time = time.time()
    user_api = internal_api.get_api()
    # api.display_in_chat(text="Starting preprocessing...", role="partial")
    # if username and chat_store_loc.get():
    #     api.datalog_to_tmp(f"New message at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if "excluded_functions" not in user_api.scope:
        user_api.scope["excluded_functions"] = ["generate_text", "get_total_tokens"]

    tracker = ConversationTracker.from_yaml(conversation_tracker_yaml)
    use_multiplexing = multiplexing.get()
    translation_val = translation_layer.get()
    if translation_val == "deepl":
        # translate last message to english for chatbot
        tracker[-1]["translated_content"] = tracker[-1]["content"]
        last_bot_msg = tracker.get_last_message(ConversationRoles.ASSISTANT)
        context = None if last_bot_msg is None else last_bot_msg
        tracker[-1]["content"] = await translate_message(tracker[-1]["content"], target_lang="EN", context=context)
    if translation_val == "LLM":
        tracker[-1]["translated_content"] = tracker[-1]["content"]
        kwargs = {"conversation_history": conversation_tracker_yaml, "to_english": True}
        translated_msg = await load_balanced_request("translate_last_message",  kwargs=kwargs)
        tracker[-1]["content"] = translated_msg

    last_usr_msg = tracker.get_last_message(ConversationRoles.USER)

    enable_knowledge_retrieval = enable_knowledge_retrieval_var.get() & enable_knowledge_retrieval

    if len(tracker.tracker)<2:
        queries = [{"query":last_usr_msg["content"],"max_entries":5}]
    else:
        convo_ctx = f"assistant: {tracker[-2]['content']}\nuser: {last_usr_msg['content']}"
        queries = [{"query":last_usr_msg["content"],"max_entries":4},
                   {"query":convo_ctx, "max_entries": 2}]
    info_score = 4
    included_functions = None
    preprocessing_tokens = None
    if use_multiplexing:
        kwargs = {"conversation_history": conversation_tracker_yaml,
                  "knowledge_retrieval_domain": knowledge_retrieval_domain,
                  "system_msg": system_msg}
        preprocessor_json, metadata = await load_balanced_request("get_preprocessing_json", kwargs=kwargs)
        preprocessing_tokens = metadata["tokens"]["total_tokens"]
        if preprocessor_json is not None:
            enable_function_calling = preprocessor_json["enable_function_calling"]
            enable_knowledge_retrieval = preprocessor_json["use_document_retrieval"]
            info_score = preprocessor_json["info_score"]
            queries = preprocessor_json["queries"]
            included_functions = preprocessor_json["included_functions"]

    context = None
    context_str = None
    if enable_knowledge_retrieval is True:
        try:
            context_str = ""
            # num_requests = len(queries)
            # maximum_entries = info_score
            # base_value = maximum_entries // num_requests
            # remainder = maximum_entries % num_requests
            # query_sizes = [base_value] * num_requests
            # for i in range(remainder):
            #     query_sizes[i] += 1
            contexts = []
            used_ids = []
            for x, query in enumerate(queries):
                # cur_query = query if not use_multiplexing else query["query"]
                cur_query = query["query"]
                print("QUERY",x, cur_query)
                future = await async_execute("query_db", args=[cur_query, knowledge_retrieval_domain, query["max_entries"]], kwargs={},
                                             return_future=True)
                context = await future
                contexts.append(context)
            for context in contexts:
                for i in context:
                    if i["id"] in used_ids:
                        continue
                    context_str += f"\n****\nID: {i['id']}\n"
                    context_str += f"DOCUMENT TITLE: {i['document_title']}\n" if "document_title" in i else ""
                    context_str += f"TITLE: {i['title']}\n" if "title" in i else ""
                    context_str += f"CONTENT: {i['content']}"
                    used_ids.append(i["id"])
            if x == 0:
                print("----DEFAULT CONTEXT----")
                print(context_str)
            if x == 1:
                print("----CONVO CONTEXT----")
                print(context_str)
            # merge the contexts into one array
            context = [item for sublist in contexts for item in sublist]
        except Exception as e:
            await api.show_message("Knowledge retrieval system faulty. No context available.", "error")
            llm_logger.exception(f"Could not retrieve context from knowledge base")
    else:
        context_str = None

    if enable_function_calling:
        if included_functions:
            user_api.scope["included_functions"] = included_functions
        func_list = _memory.get_functions_as_str(user_api.scope, short=False)
    else:
        func_list = None

    kwargs = {"conv_tracker": tracker, "context": context_str, "func_list": func_list, "system_msg": system_msg,
              "username": username,
              "chat_store_loc": chat_store_loc.get(),
              "temp": 0}

    tracker, metadata = await load_balanced_request("create_completion_plugin", kwargs=kwargs)

    assistant_msgs = tracker.pop_entry()

    all_citations = []
    total_content = ""
    code_calls = []
    msg_parts = []
    for i, msg in enumerate(assistant_msgs[::-1]):
        if "content" in msg:
            total_content += msg["content"]
            if i != len(assistant_msgs) - 1:
                total_content += "\n"
            citations = re.findall(r"\{\{(\d+)\}\}", msg["content"])#re.findall(r"\{\{([a-fA-F0-9]{16})\}\}", msg["content"])#
            try:
                citation_ids = citations
                all_citations += citation_ids
            except Exception as e:
                llm_logger.exception(f"Could not parse citation ids")
        if "code" in msg:
            if "return_value" in msg:
                code_calls.append({"code": msg["code"], "return": msg["return_value"]})
            else:
                code_calls.append({"code": msg["code"]})
        msg_parts.append(msg)
    convo_idx = 0 if len(tracker.tracker) == 0 else tracker.tracker[-1]["index"] + 1
    used_citations = []
    faulty_citations = []
    try:
        if context:
            for i in context:
                if str(i["id"]) in all_citations:
                    used_citations.append(i)
    except Exception as e:
        llm_logger.exception(f"Could not replace citations")

    for i in all_citations:
        if i not in [str(j["id"]) for j in context]:
            faulty_citations.append(i)
            total_content = re.sub(r"\{\{" + str(i) + r"\}\}", "", total_content)

    if context:
        unused_citations = [i["id"] for i in context if i["id"] not in all_citations and i["id"] not in faulty_citations]
    else:
        unused_citations = []
    for i in used_citations:
        i["index"] = int(i["id"])
        if "authors" not in i:
            i["authors"] = "Unknown"
        if "subheader" not in i:
            if "title" in i:
                i["subheader"] = i["title"]
            else:
                i["subheader"] = "Unknown"
        if "tags" not in i:
            i["tags"] = ["Unknown"]
        if type(i["tags"]) == str:
            i["tags"] = i["tags"].split(",")
    # print(used_citations)

    code_str = ""
    if len(code_calls) == 1:
        code_str = code_calls[0]["code"]
        if "return" in code_calls[0]:
            code_str += f"\nRETURN:\n{code_calls[0]['return']}"
    if len(code_calls) > 1:
        for i, code in enumerate(code_calls):
            code_str += f"CALL {i}\n{code['code']}\n"
            if "return" in code:
                code_str += f"RETURN:\n{code['return']}\n"

    merged_tracker_entry = {"role": ConversationRoles.ASSISTANT,
                            "content": total_content, }
    if len(used_citations) > 0:
        merged_tracker_entry["citations"] = used_citations
    if code_str:
        merged_tracker_entry["code"] = code_str

    merged_tracker_entry["index"] = convo_idx
    # merged_tracker_entry["metadata"] = llm.finish_meta
    # merged_tracker_entry["processing"] = preprocessor_json
    if use_multiplexing:
        multiplexing_meta_dict = {"enable_function_calling": enable_function_calling,
                                    "use_document_retrieval": enable_knowledge_retrieval,
                                    "included_functions": included_functions}
        if queries:
            multiplexing_meta_dict["queries"] = queries
            multiplexing_meta_dict["info_score"] = info_score
        multiplexing_meta_dict["preprocessing_tokens"] = preprocessing_tokens
        metadata["total_tokens"] += preprocessing_tokens
        metadata.update(multiplexing_meta_dict)
    finished_time = time.time()
    metadata["total_time"] = round(finished_time - start_time, 3)
    merged_tracker_entry["metadata"] = metadata

    if translation_val == "deepl":
        merged_tracker_entry["translated_content"] = await translate_message(merged_tracker_entry["content"],
                                                                             target_lang="DE")
    tracker.tracker.append(merged_tracker_entry)
    if translation_val == "LLM":
        kwargs = {"conversation_history": tracker.to_yaml(), "to_english": False}
        # future = await async_execute("translate_last_message", "llm_server", kwargs=kwargs, return_future=True)
        # translated_msg = await future
        translated_msg = await load_balanced_request("translate_last_message", kwargs=kwargs)
        tracker[-1]["translated_content"] = translated_msg

    tracker[-1]["metadata"]["timestamp"] = datetime.datetime.now().isoformat()
    tracker[-1]["metadata"]["model_name"] = tracker.metadata["model_name"]
    if len(unused_citations) > 0:
        tracker[-1]["metadata"]["unused_citations"] = unused_citations
    if len(all_citations)> 0:
        tracker[-1]["metadata"]["used_citations"] = all_citations
    if len(faulty_citations) > 0:
        tracker[-1]["metadata"]["faulty_citations"] = faulty_citations
    if len(msg_parts) > 1:
        tracker[-1]["sub_messages"] = yaml.dump(msg_parts)

    return tracker.to_yaml()
