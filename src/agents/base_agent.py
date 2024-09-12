# Create a base agent class that has the following functionality
#   1. think - is the generate text method (generates the response dictionary)
from abc import ABC, abstractmethod
from typing import Optional
from models.ollama_model import OllamaModel
from prompts.system_prompts import AGENT_SYS_PROMPT
# from tools.model_tools import add, multiply, subtracts


class BaseAgent(ABC):
    def __init__(
        self,
        llm,
        tools: Optional[list] = [],
        stop_token: Optional[str] = None,
        template: Optional[str] = None
    ):
        """
        Init the agent with requried model and other metadata

        Parameters:
        """
        self.llm = llm
        self.stop_token = stop_token
        self.agent_system_prompt = template or AGENT_SYS_PROMPT

        # define the tools the agent can use
        self.tools = tools

    def prepare_tools(self):
        """Prepare tool names and tool descriptions"""
        pass

    def generate_step(self, prompt: str):
        """
        This is basically the generate method
            (or calls the generate method internally)

        Parameters:
            prompt: User asked query, the question.
        """
        # prepare if there are any tools
        self.prepare_tools()

        # Add conditionals here to add support for other models
        # the message loop
        if isinstance(self.model_service, OllamaModel):
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
    def chat(self, input: str, chat_history, *args, **kwargs):
        """
        A method to format and display the response from the model.
        This method is to be defined in every agent
        """
        pass
