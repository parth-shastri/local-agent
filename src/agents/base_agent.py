# Create a base agent class that has the following functionality
#   1. chat - is the generate text method (generates the response dictionary)
import os
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Callable, Union, Literal
from src.agents import logger
from src.models.model_services import MODEL_SERVICES
from src.prompts.system_prompts import AGENT_SYSTEM_PROMPT
from src.tools.base import Tool


class BaseAgent(ABC):

    def __init__(
        self,
        model_name: str,
        model_service: Literal["ollama", "llamacpp", "groq"],
        tools: Optional[Union[Sequence[Callable], Sequence[Tool]]] = [],
        chat_format: Optional[str] = None,
        temperature: float = 0.0,
        context_window: int = 4096,
        stop_token: Optional[str] = None,
        system_prompt: Optional[str] = None,
        is_tool_use_model: bool = True,
        **generation_kwargs,
    ):
        """
        Init the agent with requried model and other metadata

        Parameters:
        """
        self.model_name = model_name
        self.chat_format = chat_format
        self.model_service = model_service
        self.agent_system_prompt = (system_prompt or AGENT_SYSTEM_PROMPT) if not is_tool_use_model else system_prompt
        self.is_tool_use_model = is_tool_use_model

        # llm arguments
        self.temperature = temperature
        self.context_window = context_window
        self.stop_token = stop_token
        self.generation_kwargs = generation_kwargs or {}

        # define the tools the agent can use
        self.tools = tools
        self.tool_dict = dict(map(lambda x: (x.tool_name, x), self.tools))

        # init the LLM
        self.llm = self._init_llm()
        # Format the system prompt
        self._format_system_prompt()

    @property
    def system_prompt(self):
        """The formatted system prompt"""
        return self.agent_system_prompt

    def _format_system_prompt(self):
        """Format the system prompt"""
        # only format if not natively supported
        if not self.is_tool_use_model:
            self.agent_system_prompt.format(
                tool_descriptions=[tool.tool_description for tool in self.tools]
            )

    def _init_llm(self):
        try:
            # define the kwargs (are needed for the LlamaCPP model service)
            kwargs = dict(
                n_threads=os.cpu_count() - 4,
                chat_format=self.chat_format
            )
            service = MODEL_SERVICES[self.model_service]
            llm = service(
                model=self.model_name,
                system_prompt=self.agent_system_prompt,
                temperature=self.temperature,
                context_window=self.context_window,
                stop=self.stop_token,
                is_tool_use_model=self.is_tool_use_model,
                **self.generation_kwargs,
                **kwargs,
            )
            return llm
        except KeyError as e:
            logger.error(e)
            raise ValueError(
                f"Can only serve locally with [ollama, llamacpp] currently, found model_service={self.model_service}"
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

    def generate_step(self, prompt: str, json_mode=False):
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
            model_instance = MODEL_SERVICES["ollama"](
                model=self.model_name,
                system_prompt=self.agent_system_prompt,
                temperature=self.temperature,
                context_window=self.context_window,
                stop=self.stop_token,
                json_mode=json_mode,
                is_tool_use_model=self.is_tool_use_model,
                **self.generate_kwargs
            )

        else:
            raise ValueError(
                f"Can only serve locally with ollama currently, found model_service={self.model_service}"
            )

        # get the response dict from the model_instance
        response_dict = model_instance.generate_text_api_call(prompt)

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
