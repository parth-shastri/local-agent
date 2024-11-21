# A class to query the ollama service running on the localhost
import json
import os
from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from src.tools.base import Tool
from src.models.base_model import BaseLLM
from typing import Sequence, Type, Optional, Union, Dict, Any
from pydantic import Field, PrivateAttr
from termcolor import colored


class LlamaCPPModel(BaseLLM):
    """
    Llama CPP model.

    Install llama-cpp-python following instructions:
        https://github.com/abetlen/llama-cpp-python

    """

    model: str = Field(
        description="The repo_id (Hugging Face), model_path for the Model"
    )
    model_file: str = Field(
        description="The model_file to use from within the repo / model_path"
    )
    temperature: float = Field(
        default=0.1,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    context_window: int = Field(
        default=4096,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    is_tool_use_model: bool = Field(
        default=False, description="Whether the model is a function calling model."
    )
    generation_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Kwargs used for generation, incl parameters like topK, temperature etc..",
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Kwargs used for model initialization, incl parameters like, Lora params, n_gpus, context parameters.",
    )
    verbose: bool = Field(
        default_factory=False, description="Display the internals of the model calls."
    )

    _client: Optional[Llama] = PrivateAttr()

    def __init__(
        self,
        model: str,
        system_prompt,
        temperature=0.1,
        context_window: int = 4096,
        max_new_tokens: Optional[int] = None,
        stop=None,
        is_tool_use_model=True,
        verbose: Optional[bool] = False,
        generation_kwargs: Optional[dict[str, Any]] = None,
        **kwargs
    ):
        """
        Init the LlamaCPP model with the given parameters

        Parameters:
            model (str): The the `repo_id/model_name` of the model to use.
            system_prompt (str): The system prompt to use.
            temperature (float): The temperature setting for the model.
            stop (str): The stop token for the model.
        """
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            context_window=context_window,
            stop=stop,
            is_tool_use_model=is_tool_use_model,
            verbose=verbose,
        )
        # override the model parameters.
        self.model, self.model_name = model.split("/")
        # look for the chat format on the llamacpp docs: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama
        self.chat_format = kwargs.pop("chat_format", 'functionary-v2')
        self.n_threads = kwargs.pop("n_threads", os.cpu_count() - 4)
        # Args incl model loading args like, lora params, GPU parameters etc
        self.model_kwargs = {
            **{"n_ctx": self.context_length, "verbose": verbose},
        }
        # Args incl top_p, top_k, stop etc.
        self.generation_kwargs = {
            **{
                "temperature": self.temperature,
                "max_tokens": max_new_tokens,
                "stop": self.stop,
            }
        }
        # override
        self.generation_kwargs.update(generation_kwargs or {})
        # init client
        self._client = None
        self._init_client()

    @property
    def tokenizer(self):
        try:
            return LlamaHFTokenizer.from_pretrained(self.model)
        except OSError:
            raise ValueError(
                "Invalid model: No model path satisfies the given model, please make sure the model is a valid HuggingFace repo."
            )

    @property
    def client(self):
        """Property to access the client directly (chat, generate etc.)"""
        return self._client

    def _init_client(self):
        # currently only supports pretrained models from hugging face.
        try:
            self._client = Llama.from_pretrained(
                repo_id=self.model,
                filename=self.model_name,
                chat_format=self.chat_format,
                chat_handler=(
                    Jinja2ChatFormatter(
                        template=self.tokenizer.hf_tokenizer.chat_template,
                        eos_token=self.tokenizer.hf_tokenizer.eos_token,
                        bos_token=self.tokenizer.hf_tokenizer.bos_token,
                    ).to_chat_handler()
                    if not self.chat_format
                    else None
                ),
                response_format="json_object",
                tokenizer=self.tokenizer,
                # load all the layers on the GPU
                n_gpu_layers=-1,
                # thread utilization for cpu model
                n_threads=self.n_threads,
                **self.model_kwargs,
            )
        except Exception as e:
            raise ValueError(
                "Check the model_path / model_name provided"
                "Check the chat format provided"
                f"{e}"
            )

    def chat(
        self,
        input: Union[str, dict[str, str]],
        chat_history: Optional[Sequence[dict]] = None,
        tools: Optional[Sequence[Type[Tool]]] = None,
        json_mode: bool = False
    ):
        """
        Chat with the model
         Does the tool call and returns the output
         Validates the model response to follow the intended schema.

         Example response structure ferom the LlamaCPP client:
         ```
         {'id': 'chatcmpl-...',
         'object': 'chat.completion',
         'created': 1726547058,
         'model': '/home/ostrich/.cache/huggingface/hub/...
         'choices': [{'index': 0,
         'logprobs': None,
         'message': {'role': 'assistant',
         'content': ...},
         'finish_reason': 'stop'}],
         'usage': {'prompt_tokens': 537, 'completion_tokens': 17, 'total_tokens': 551}}
         ```
        """
        # format the messages according to requirement
        messages = self.convert_messages(input, chat_history)

        if self.verbose:
            print(f"Input: {messages}")

        # create a tool_dict to map the called_tool back to tools
        tools = tools or []
        tool_dict = dict(map(lambda x: (x.tool_name, x), tools))

        if self.is_tool_use_model:
            client_response = self.client.create_chat_completion(
                messages=messages,
                tools=[tool.to_openai_tool() for tool in tools],
                tool_choice="auto",
                response_format={"type": "json_object"} if json_mode else None,
                **self.generation_kwargs,
            )
            if self.verbose:
                print(colored(f"\n[MODEL]: {client_response}\n", color="light_yellow"))
            model_response = client_response["choices"][0]["message"]
            # get the tool_call response & extract the tool name
            # Notify the user if no tool is used.
            tool_response = model_response.get("tool_calls", [])

            # if the tool response is None
            if not tool_response:
                response = model_response
                response[
                    "content"
                ] += "\n**Disclaimer: The output was generated without using any tools."
                return response

            # Iterate through all the tools calls & validate
            for tool in tool_response:
                # ===== This chunk of code is validating the tool response against schema ===
                tool_name = tool["function"]["name"]
                # get the function args to call the function later.
                tool_arguments = tool["function"]["arguments"]
                # validate the response
                called_tool = tool_dict[tool_name]
                args = self._validate_structured_response(
                    response=tool_arguments, called_tool=called_tool
                )
            response = model_response
        else:
            client_response = self.client.create_chat_completion(
                messages=messages,
                tools=[tool.to_openai_tool() for tool in tools],
                tool_choice="auto",
                response_format={"type": 'json_object'},
                **self.generation_kwargs,
            )
            if self.verbose:
                print(colored(f"\n[MODEL]: {client_response}\n", color="light_yellow"))
            # get the model_response
            model_response = client_response["choices"][0]["message"]
            # get the message content
            content = model_response["content"]

            # The case if the model is not a tool-call supported model on ollama but..
            # ..we have specified the proper system prompt
            if tools:
                # parse content to json .loads ?
                content = json.loads(content)
                # logic to get the tool call from the response
                tool_response = content.get("tool_input")
                tool_name = content.get("tool_choice")
                # get the called tool
                called_tool = tool_dict[tool_name]
                # validate the response
                args = self._validate_structured_response(
                    response=tool_response, called_tool=called_tool
                )
                # create a client_response['message'] structure.
                response = {
                    "role": model_response["role"],
                    "content": model_response["content"],
                    "tool_calls": [
                        {"function": {"name": tool_name, "arguments": args}}
                    ],
                }

            response = model_response
        return response
