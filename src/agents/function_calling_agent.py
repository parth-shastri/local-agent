"""
A simple function calling agent
"""

from src.agents.base_agent import BaseAgent
from typing import Literal, Optional, Union, Sequence, Callable, Any
from src.tools.base import Tool
from pydantic import ValidationError
from termcolor import colored


class FunctionCallingAgent(BaseAgent):

    def __init__(
        self,
        model_name: str,
        model_service: Literal["ollama"],
        tools: Optional[Union[Sequence[Callable], Sequence[Tool]]] = [],
        temperature: float = 0.0,
        context_window: int = 4096,
        stop_token: Optional[str] = None,
        system_prompt: Optional[str] = None,
        is_tool_use_model: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            model_service=model_service,
            tools=tools,
            temperature=temperature,
            context_window=context_window,
            stop_token=stop_token,
            system_prompt=system_prompt,
            is_tool_use_model=is_tool_use_model,
        )

    def _call_function(self, tool_call: Tool, response: Any):
        """Call the tool function and get the final response"""
        # given the tool_call & tool_name get the tool response
        args = response["tool_calls"][0]["function"]['arguments']
        output = tool_call.tool_output(args)
        # return type like the openai messages format.
        return {
            "role": "tool",
            "name": tool_call.tool_name,
            "content": str(output),
            "return_direct": tool_call.return_direct,
            "tool_response": response
        }

    def finalize_response(self, input, chat_history, tool_output):
        """Finalize the response of the tool call"""
        # call the chat completion method using the chat history
        # add input & tool_response to chat history
        # NOTE: Is this even right? Do I need to pass the whole chat history for the final response?
        input_message = {"role": "user", 'content': input}
        if chat_history is not None:
            chat_history.append(input_message)
        else:
            chat_history = [input_message]

        if model_tool_reponse := tool_output.get("tool_response", None):
            chat_history.append(model_tool_reponse)
        else:
            raise ValueError("Didn't get the function calling response by the model")

        tool_message = {"role": tool_output["role"], "name": tool_output["name"], "content": tool_output["content"]}

        # print(tool_message, chat_history)

        response = self.llm.chat(tool_message, chat_history, tools=None)
        return response['content']

    def run_step(self, prompt: str, chat_history: Optional[Sequence[dict]] = None, use_tools: bool = True):
        """Run one execution step"""
        # Validate chat history
        if chat_history:
            self._validate_chat_history(chat_history)

        if use_tools:
            response = self.llm.chat(prompt, chat_history, tools=self.tools)
        else:
            response = self.llm.chat(prompt, chat_history, tools=[])

        return response

    def chat(self, input: str, chat_history: Optional[Sequence[dict]] = None, max_retries: int = 3, verbose=False, *args, **kwargs):
        """Chat using input"""
        trial_no = 0
        final_response = None
        while trial_no <= max_retries + 1:  # The first try will be excluede
            # Run the execution loop only until we reach max retries
            try:
                response = self.run_step(input, chat_history, use_tools=True)

                if verbose:
                    print(colored(f"[AGENT]: Model Response: {response}", color='light_green'))
                # if the name field of the response is not Empty
                if tool_calls := response.get('tool_calls', []):
                    # call the tool
                    tool_name = tool_calls[0]['function']['name']
                    tool_output = self._call_function(self.tool_dict[tool_name], response)

                    # either return the response as it is or finalize it before returning
                    final_response = self.finalize_response(input, chat_history, tool_output) if not tool_output['return_direct'] else tool_output['content']

                    return final_response

                else:
                    final_response = response["content"]
                    return final_response

            except ValidationError as e:
                # retry the execution
                if verbose:
                    print(colored(f"[AGENT]: Encountered Validation Error, retrying execution: {e}", color='light_red'))
                trial_no += 1
                continue

            except KeyError as e:
                # Fallback
                # This error means an unspecified tool was called
                # Small models generally have this problem
                # call the run_step method with use_tools=False to answer from model_knowledge.
                if verbose:
                    print(colored(f"[AGENT]: Encountered KeyError: {e}", color='light_red'))
                    print(colored("Attempting to Answer from model Knowledge !", color='cyan'))
                additional_context = " Answer from your knowledge & provide a disclaimer to the user."
                response = self.run_step(input + additional_context, chat_history, use_tools=False)
                final_response = response['content']
                return final_response

        # if we come out of the loop unreturned it is certain that we have exceeded the max_tries.
        if final_response is None and trial_no >= max_retries:
            raise Exception("MaxRetries exceeded error!: Reached the max retries for the agent")

        return final_response
