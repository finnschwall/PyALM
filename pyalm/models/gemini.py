from pyalm.internal.alm import ALM
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
from timeit import default_timer as timer


class Gemini(ALM):
    """
    PyALM implementation for Gemini. Requires the Vertex AI SDK to be installed.
    """

    def __init__(self, model_path_or_name="gemini-pro", project=None, verbose=0, location="europe-west3",**kwargs):
        super().__init__(model_path_or_name, verbose=verbose)
        vertexai.init(project=project, location=location)
        self.model = GenerativeModel(model_path_or_name)
        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        openai_specifics = {"ASSISTANT": "output", "USER": "input", "SYSTEM": ""}
        self._built_in_symbols.update(openai_specifics)
        self.settings.include_conv_id_as_stop = False

    def build_prompt(self, preserve_flow=False):
        return self.build_prompt_as_str(1, 0, block_gen_prefix=preserve_flow)

    def tokenize(self, text):
        raise Exception("No offline tokenizer for gemini available")

    def tokenize_as_str(self, text):
        raise Exception("No offline tokenizer for gemini available")

    # TODO add static variant
    def get_n_tokens(self, text):
        self.model.count_tokens(text)

    def create_native_completion(self, text, max_tokens=256, stop=None, keep_dict=False, token_prob_delta=None,
                                 token_prob_abs=None,
                                 log_probs=None, **kwargs):
        start = timer()
        add_keys = {"stop_sequences": stop if isinstance(stop,list) else [stop]} if stop else {}

        responses = self.model.generate_content(
            text,
            generation_config={
                "max_output_tokens": max_tokens, **add_keys, **kwargs
            },
            safety_settings=self.safety_settings,
            stream=False,

        )
        end = timer()
        resp_dict = responses.to_dict()
        response = resp_dict["candidates"][0]
        self.finish_meta["finish_reason"] = response["finish_reason"]
        usage_metadata = resp_dict["usage_metadata"]
        tok_in = usage_metadata["prompt_token_count"]
        tok_gen = usage_metadata["candidates_token_count"]
        tok_total = usage_metadata["total_token_count"]

        self.finish_meta["tokens"] = {"prompt_tokens": tok_in, "generated_tokens": tok_gen, "total_tokens": tok_total}
        self.finish_meta["timings"] = {"total_time": round(end - start, 3)}
        self.finish_meta["t_per_s"] = {"token_total_per_s": tok_total / (end - start)}
        self.finish_meta["safety_ratings"] = response

        if keep_dict:
            return response
        return response["content"]["parts"][0]["text"]


    def _extract_message_from_generator(self, gen):

        for i in gen:
            try:
                i = i.to_dict()
                token = i["candidates"][0]["content"]["parts"][0]["text"]
                if "usage_metadata" in i:
                    self.finish_meta["finish_reason"] = i["candidates"][0]["finish_reason"]
                    usage_metadata = i["usage_metadata"]
                    tok_in = usage_metadata["prompt_token_count"]
                    tok_gen = usage_metadata["candidates_token_count"]
                    tok_total = usage_metadata["total_token_count"]
                    self.finish_meta["tokens"] = {"prompt_tokens": tok_in, "generated_tokens": tok_gen,
                                                  "total_tokens": tok_total}
                    self.finish_meta["safety_ratings"] = i["candidates"][0]["safety_ratings"]
                if token is None:
                    break
                yield token, None
            except Exception as e:
                print(e)
                pass

    def create_native_generator(self, text, keep_dict=False, token_prob_delta=None,
                                token_prob_abs=None, **kwargs):
        # TODO make this class function
        rename_mapping = {"max_tokens": "max_output_tokens"}
        renamed_dict = {}
        for old_key, new_key in rename_mapping.items():
            if old_key in kwargs:
                renamed_dict[new_key] = kwargs[old_key]
            else:
                pass

        responses = self.model.generate_content(
            text,
            generation_config={
                **renamed_dict
            },
            safety_settings=self.safety_settings,
            stream=True,
        )

        if keep_dict:
            return responses
        else:
            return self._extract_message_from_generator(responses)
