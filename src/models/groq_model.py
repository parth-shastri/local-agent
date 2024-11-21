# A class to query the ollama service running on the localhost
import json
from groq import Groq
from src.tools.base import Tool
from src.models.base_model import BaseLLM
from typing import Sequence, Type, Optional, Union, Dict, Any
from pydantic import Field, PrivateAttr
from termcolor import colored


class GroqModel(BaseLLM):
    """
    Groq model.

    Install groq model using `pip install groq`

    """

    model: str = Field(
        description="The repo_id (Hugging Face), model_path for the Model"
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
    request_timeout: float = Field(
        default=120.0,
        description="The timeout for making http request to Ollama API server",
    )
    is_tool_use_model: bool = Field(
        default=False, description="Whether the model is a function calling model."
    )
    generation_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Kwargs used for generation, incl parameters like topK, temperature etc..",
    )
    _client: Optional[Groq] = PrivateAttr()

    def __init__(
        self,
        model: str,
        system_prompt,
        temperature=0.1,
        context_window: int = 4096,
        max_new_tokens: Optional[int] = None,
        stop=None,
        request_timeout: int = 120.0,
        is_tool_use_model=True,
        verbose: Optional[bool] = False,
        generation_kwargs: Optional[dict[str, Any]] = None,
        **kwargs
    ):
        """
        Init the Groq model with the given parameters

        Parameters:
            model (str): The repo id of the model to use
            model_name (str): The model filename to use from within the repo
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
            verbose=verbose
        )

        self.headers = {"Content-Type": "application/json"}
        self.request_timeout = request_timeout
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

    def _init_client(self):
        """Property to access the client directly (chat, generate etc.)"""
        self._client = Groq(timeout=self.request_timeout, default_headers=self.headers)

    @property
    def client(self):
        """Property to access the client directly (chat, generate etc.)"""
        return self._client

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

         Example response structure from the Groq client:
         ```
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
            client_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[tool.to_openai_tool() for tool in tools],
                tool_choice="auto",
                response_format={"type": "json_object"} if json_mode else None,
                **self.generation_kwargs,
            )
            print(colored(f"\n[MODEL]: {client_response}\n", color="light_yellow"))
            model_response = client_response.to_dict()['choices'][0]['message']
            # get the tool_call response & extract the tool name
            # Notify the user if no tool is used.
            tool_response = model_response.get("tool_calls", [])

            # if the tool response is None
            # TODO: remove the disclaimer once the dev is complete
            if not tool_response:
                response = model_response
                response[
                    "content"
                ] += "\n**Disclaimer: The output was generated without using any tools."
                return response

            # handle all the tool calls
            for tool in tool_response:
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
            client_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[tool.to_openai_tool() for tool in tools],
                tool_choice="auto",
                response_format={"type": "json_object"},
                ** self.generation_kwargs,
            )
            print(colored(f"\n[MODEL]: {client_response}\n", color="light_yellow"))
            # get the model_response
            model_response = client_response.to_dict()["choices"][0]["message"]
            # get the message content
            content = model_response["content"]

            # The case if the model is not a tool-call supported model but..
            # ..we have specified the proper system prompt
            # parse content to json .loads ?
            content = json.loads(content)
            # logic to get the tool call from the response
            tool_response = content.get("tool_input")
            tool_name = content.get("tool_choice")

            if tools and tool_name:
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

            response = {"role": model_response["role"], "content": tool_response}
        return response
