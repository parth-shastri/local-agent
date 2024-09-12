# A class to query the ollama service running on the localhost
import requests
import json
from ollama import Client
from tools.base import Tool
from typing import List, Sequence, Type, Optional, Union
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
                self.model.generate_endpoint,
                headers=self.headers,
                data=json.dumps(payload),
            )
            print(f"REQUEST RESPONSE: {request_response}")
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
        input: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
    ):
        """Convert messages intp OpenAI format (This is what ollama accepts)"""
        input_message = {"role": "user", "content": input}
        if system_prompt is not None:
            system_message = {"role": "system", "content": system_prompt}

        messages = [system_message, input_message]

        # add history if given
        if chat_history is not None:
            messages = chat_history.extend(messages)
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
        input: str,
        system_prompt: str,
        chat_history: List,
        tools: Optional[Sequence[Type[Tool]]] = None,
    ):
        """Chat with the model"""
        # format the messages according to requirement
        messages = self.convert_messages(input, system_prompt, chat_history)

        # create a tool_dict to map the called_tool back to tools
        tool_dict = dict(map(lambda x: (x.tool_name, x), tools))

        if self.is_tool_use_model:
            client_response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                format="json" if self.json_mode else "",
            )
            # get the tool_call response & extract the tool name
            tool_response = client_response["message"].get("tool_calls", [])
            tool_name = tool_response["function"]["name"]
            # get the function args to call the function later.
            tool_arguments = tool_response["function"]["arguments"]
            # validate the response
            called_tool = tool_dict[tool_name]
            args = self._validate_structured_response(
                response=tool_arguments, called_tool=called_tool
            )
            # call the tool if return_direct is not None
            response = called_tool.call(args)

        else:
            client_response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=None,
                format="json" if self.json_mode else "",
            )
            # get the message content
            response = client_response["message"]["content"]

            # The case if the model is not a tool-call supported model on ollama but
            # we have specified the proper system prompt
            if tools is not None:
                # logic to get the tool call from the response
                tool_response = response.get("tool_call")
                tool_name = response.get("tool_name")
                # get the called tool
                called_tool = tool_dict[tool_name]
                # validate the response
                args = self._validate_structured_response(
                    response=tool_response, called_tool=called_tool
                )
                # call the tool
                response = called_tool.call(args)

        return response
