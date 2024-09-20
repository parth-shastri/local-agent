# A class to query the ollama service running on the localhost
import requests
import json
import ollama
from ollama import Client
from src.tools.base import Tool
from src.models.base_model import BaseLLM
from typing import Sequence, Type, Optional, Union, Dict, Any
from pydantic import Field, PrivateAttr
from termcolor import colored


class OllamaModel(BaseLLM):
    """
    Ollama models served through the ollama endpoint

    Visit https://ollama.com/ to download and install Ollama.

    Run `ollama serve` to start a server.

    Run `ollama pull <name>` to download a model to run.
    """

    model: str = Field(description="The Ollama model to use.")
    temperature: float = Field(
        default=0.75,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    context_window: int = Field(
        default=4096,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    request_timeout: float = Field(
        default=120.0,
        description="The timeout for making http request to Ollama API server",
    )
    json_mode: bool = Field(
        default=True, description="Whether to use the JSON model of the OllamaAPI"
    )
    is_tool_use_model: bool = Field(
        default=False, description="Whether the model is a function calling model."
    )
    url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted under.",
    )

    generation_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model generation parameters for the Ollama API.",
    )

    verbose: bool = Field(
        default_factory=False, description="Display the internals of the model calls."
    )

    _client: Optional[Client] = PrivateAttr()

    def __init__(
        self,
        model,
        system_prompt,
        temperature=0,
        context_window: int = 4096,
        stop=None,
        url="http://localhost:11434",
        json_mode: bool = False,
        is_tool_use_model=True,
        verbose=False,
        # optional
        generation_kwargs: Optional[dict[str, Any]] = None
    ):
        """
        Init the OllamaModel with the given parameters

        Parameters:
            model (str): The name of the model to use.
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
            json_mode=json_mode,
            is_tool_use_model=is_tool_use_model,
            verbose=verbose
        )
        self.url = url
        self.model_generate_endpoint = url + "/api/generate"
        self.headers = {"Content-Type": "application/json"}
        self._client = None
        # specify the generation kwargs
        self.generation_kwargs = {
            **{"temperature": self.temperature},
        }
        # override the arguments
        self.generation_kwargs.update(generation_kwargs or {})
        # check for the model_name
        self._check_model_name()

    def _check_model_name(self):
        """Check if the given model name is served through ollama"""
        model_list = map(lambda x: x["name"], ollama.list()['models'])
        if self.model not in model_list:
            raise ValueError(f"The Ollama model not found locally, found {model_list} models try one of these or pull the model by `ollama pull {self.model_name}`")

    @property
    def client(self):
        """Property to access the client directly (chat, generate etc.)"""
        self._client = Client(host=self.url, timeout=120.0)
        return self._client

    def generate_text_api_call(self, prompt):
        """
        Generates response from the Ollama model, based on the provided prompt.
        """
        payload = {
            "model": self.model,
            "format": "json" if self.json_mode else "",
            "prompt": prompt,
            "system_prompt": self.system_prompt,
            "stream": False,
            "temperature": self.temperature,
            "stop": self.stop,
        }

        try:
            request_response = requests.post(
                self.model_generate_endpoint,
                headers=self.headers,
                data=json.dumps(payload),
            )
            if self.verbose:
                print(f"[MODEL]: REQUEST RESPONSE: {request_response.status_code}")
            request_response_json = request_response.json()
            response = request_response_json["response"]
            if self.verbose:
                print(f"\n\nResponse from OllamaModel::{self.model}={response}")

            return response

        except requests.RequestException as e:
            response = {"error": f"Error in invoking the model: {str(e)}"}

            return response

    def chat(
        self,
        input: Union[str, dict[str, str]],
        chat_history: Optional[Sequence[dict]] = None,
        tools: Optional[Sequence[Type[Tool]]] = None,
    ):
        """
        Chat with the model
            Does the tool call and returns the output
            Validates the model response to follow the intended schema.
        """
        # format the messages according to requirement
        messages = self.convert_messages(input, chat_history)
        if self.verbose:
            print(f"Input: {messages}")

        # create a tool_dict to map the called_tool back to tools
        tools = tools or []
        tool_dict = dict(map(lambda x: (x.tool_name, x), tools))

        if self.is_tool_use_model:
            client_response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=[tool.to_openai_tool() for tool in tools],
                format="json" if self.json_mode else "",
            )
            if self.verbose:
                print(colored(f"\n[MODEL]: {client_response}\n", color="light_yellow"))
            # model response
            model_response = client_response['message']
            # get the tool_call response & extract the tool name
            # Notify the user if no tool is used.
            tool_response = model_response.get("tool_calls", [])

            # if the tool response is None
            if not tool_response:
                response = model_response
                response[
                    "content"
                ] += "\nDisclaimer: The output was generated without using any tools."
                return response

            tool_name = tool_response[-1]["function"]["name"]
            # get the function args to call the function later.
            tool_arguments = tool_response[-1]["function"]["arguments"]
            # validate the response
            called_tool = tool_dict[tool_name]
            args = self._validate_structured_response(
                response=tool_arguments, called_tool=called_tool
            )
            response = model_response
        else:
            client_response = self.client.chat(
                model=self.model, messages=messages, tools=None, format="json"
            )
            if self.verbose:
                print(colored(f"\n[MODEL]: {client_response}\n", color="light_yellow"))
            # get the model response
            model_response = client_response['message']
            # get the message content
            content = model_response["content"]

            # The case if the model is not a tool-call supported model on ollama but
            # we have specified the proper system prompt
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
