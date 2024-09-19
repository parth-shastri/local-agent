from src.tools.base import Tool
from typing import Sequence, Type, Optional, Union
from abc import ABC, abstractmethod
from pydantic import ValidationError


class BaseLLM(ABC):

    def __init__(
        self,
        model,
        system_prompt,
        temperature=0,
        context_window: int = 4096,
        stop=None,
        json_mode: bool = False,
        is_tool_use_model=True,
        verbose: bool = False
    ):
        self.temperature = temperature
        self.context_length = context_window
        self.model = model
        self.system_prompt = system_prompt

        self.stop = stop
        self.json_mode = json_mode
        self.is_tool_use_model = is_tool_use_model
        self.verbose = verbose

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
        elif isinstance(input, dict) and (
            input.get("role", None) and input.get("content", None)
        ):
            # TODO: add validation logic here, make a type for the input message
            input_message = input
        else:
            raise ValueError(
                "Input message is not according to the expected format, Expected: {'role': , 'content'}"
            )

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

    @abstractmethod
    def chat():
        """The chat interface of the model with / without tool use"""
        pass
