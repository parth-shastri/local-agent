# Create a base agent class that has the following functionality
#   1. think - is the generate text method (generates the response dictionary)
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Callable, Union, Literal
from src.models.ollama_model import OllamaModel
from src.prompts.system_prompts import AGENT_SYSTEM_PROMPT
from src.tools.base import Tool


class BaseAgent(ABC):

    def __init__(
        self,
        model_name: str,
        model_service: Literal['ollama'],
        tools: Optional[Union[Sequence[Callable], Sequence[Tool]]] = [],
        stop_token: Optional[str] = None,
        system_prompt: Optional[str] = None,
        is_tool_use_model: bool = True,
    ):
        """
        Init the agent with requried model and other metadata

        Parameters:
        """
        self.model_name = model_name
        self.model_service = model_service
        self.stop_token = stop_token
        self.agent_system_prompt = system_prompt or AGENT_SYSTEM_PROMPT
        self.is_tool_use_model = is_tool_use_model

        # define the tools the agent can use
        self.tools = tools
        self.tool_dict = dict(map(lambda x: (x.tool_name, x), self.tools))

        # init the LLM
        self.llm = self._init_llm()

    def _init_llm(self):
        if self.model_service == "ollama":
            llm = OllamaModel(
                model=self.model_name,
                system_prompt=self.agent_system_prompt,
                stop=self.stop_token,
                is_tool_use_model=self.is_tool_use_model
            )
            return llm
        else:
            raise ValueError(
                f"Can only serve locally with ollama currently, found model_service={self.model_service}"
            )

    def _validate_chat_history(self, chat_history: Sequence[dict]):
        """validate chat history"""
        for message in chat_history:
            # check if the message has any role other than 'user', 'assistant', 'tool'
            if message["role"] not in ["user", "assistant", "tool"]:
                raise ValueError(
                    "Chat history can only have messages from ['user', 'assistant', 'tool]",
                    "Chat history can't have messages from 'system'",
                )

    def prepare_tools(self):
        """Prepare tool names and tool descriptions"""
        if self.tools and not isinstance(self.tools[0], Tool):
            self.tools = map(lambda x: Tool.from_function(function=x))

    def generate_step(self, prompt: str):
        """
        This is basically the generate method
        One-time generation.
        Doesn't take history into account.

        Parameters:
            prompt: User asked query, the question.
        """
        # prepare if there are any tools
        self.prepare_tools()

        # Add conditionals here to add support for other models
        # the message loop
        if self.model_service == "ollama":
            model_instance = self.model_service(
                model=self.model_name,
                system_prompt=self.agent_system_prompt,
                temperature=0,
                stop=self.stop_token,
            )

        else:
            raise ValueError(
                f"Can only serve locally with ollama currently, found model_service={self.model_service}"
            )

        # get the response dict from the model_instance
        response_dict = model_instance.generate_text(prompt)

        return response_dict

    @abstractmethod
    def run_step(self, prompt: str, chat_history):
        """run an execution step for the agent"""
        pass

    @abstractmethod
    def chat(self, input: str, chat_history, *args, **kwargs):
        """
        A method to format and display the response from the model.
        This method is to be defined in every agent
        """
        pass
