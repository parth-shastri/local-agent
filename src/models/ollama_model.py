# A class to query the ollama service running on the localhost
import requests
import json
from ollama import Client
from src.tools.base import Tool
from typing import Sequence, Type, Optional, Union
from pydantic import ValidationError


class OllamaModel:
    "Ollama models served through the ollama endpoint"

    def __init__(
        self,
        model,
        system_prompt,
        temperature=0,
        context_length: int = 3900,
        stop=None,
        url="http://localhost:11434",
        json_mode: bool = False,
        is_tool_use_model=True,
    ):
        """
        Init the OllamaModel with the given parameters

        Parameters:
            model (str): The name of the model to use.
            system_prompt (str): The system prompt to use.
            temperature (float): The temperature setting for the model.
            stop (str): The stop token for the model.
        """
        self.url = url
        self.model_generate_endpoint = url + "/api/generate"
        self.temperature = temperature
        self.context_length = context_length
        self.model = model
        self.system_prompt = system_prompt
        self.stop = stop
        self.headers = {"Content-Type": "application/json"}
        self.json_mode = json_mode
        # is tool use available for model
        self.is_tool_use_model = is_tool_use_model
        self._client = None

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
            "format": "json",
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
            print(f"REQUEST RESPONSE: {request_response.status_code}")
            request_response_json = request_response.json()
            response = request_response_json["response"]
            response_dict = json.loads(response)

            print(f"\n\nResponse from OllamaModel::{self.model}={response_dict}")

            return response_dict

        except requests.RequestException as e:
            response = {"error": f"Error in invoking the model: {str(e)}"}

            return response

    def convert_messages(
        self,
        input: Union[str, Sequence[dict]],
        chat_history: Optional[Sequence[dict]] = None,
    ):
        """
        Convert messages intp OpenAI format (This is what ollama accepts)
            Order: sys_message - chat_history - input
        """
        messages = []
        # input message handling.
        if isinstance(input, str):
            input_message = {"role": "user", "content": input}
        elif isinstance(input, dict) and (input.get('role', None) and input.get('content', None)):
            # TODO: add validation logic here, make a type for the input message
            input_message = input
        else:
            raise ValueError("Input message is not according to the expected format, Expected: {'role': , 'content'}")

        # system prompt handling
        if self.system_prompt is not None:
            system_message = {"role": "system", "content": self.system_prompt}
            messages.append(system_message)

        # add history if given
        if chat_history is not None:
            messages.extend(chat_history)
        # add the input message to messages
        messages.append(input_message)
        return messages

    def _validate_structured_response(
        self, response: Union[str, dict], called_tool: Type[Tool]
    ):
        "Validation of the tool response of the model"
        try:
            tool_schema = called_tool.tool_schema
            # validate the tool_schema
            if isinstance(response, str):
                parsed_response = tool_schema.model_validate_json(response)
            elif isinstance(response, dict):
                parsed_response = tool_schema.model_validate(response)

            return parsed_response.model_dump()

        except ValidationError as e:
            raise e

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
        print(messages)

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
            print(client_response)
            # get the tool_call response & extract the tool name
            tool_response = client_response["message"].get("tool_calls", [])
            # if the tool response is None
            if not tool_response:
                response = client_response['message']
                return response

            tool_name = tool_response[0]["function"]["name"]
            # get the function args to call the function later.
            tool_arguments = tool_response[0]["function"]["arguments"]
            # validate the response
            called_tool = tool_dict[tool_name]
            args = self._validate_structured_response(
                response=tool_arguments, called_tool=called_tool
            )
            response = client_response["message"]
        else:
            client_response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=None,
                format="json" if self.json_mode else "",
            )
            # get the message content
            content = client_response["message"]["content"]

            # The case if the model is not a tool-call supported model on ollama but
            # we have specified the proper system prompt
            if tools:
                # logic to get the tool call from the response
                tool_response = content.get("tool_call")
                tool_name = content.get("tool_name")
                # get the called tool
                called_tool = tool_dict[tool_name]
                # validate the response
                args = self._validate_structured_response(
                    response=tool_response, called_tool=called_tool
                )
                # create a client_response['message'] structure.
                response = {
                    "role": client_response['message']['role'],
                    "content": client_response["message"]['content'],
                    "tool_calls": [{"function": {"name": tool_name, 'arguments': args}}]
                }

            response = client_response['message']
        return response
