"""
A simple function calling agent
"""
from src.agents.base_agent import BaseAgent
from typing import Literal, Optional, Union, Sequence, Callable, Any
from src.tools.base import Tool
from pydantic import ValidationError
from termcolor import colored
from src.agents import logger


class FunctionCallingAgent(BaseAgent):

    def __init__(
        self,
        model_name: str,
        model_service: Literal["ollama", "llamacpp", "groq"],
        tools: Optional[Union[Sequence[Callable], Sequence[Tool]]] = [],
        model_path: Optional[str] = None,
        chat_format: Optional[str] = None,
        temperature: float = 0.0,
        context_window: int = 4096,
        stop_token: Optional[str] = None,
        system_prompt: Optional[str] = None,
        is_tool_use_model: bool = True,
        **generation_kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_service=model_service,
            tools=tools,
            model_path=model_path,
            chat_format=chat_format,
            temperature=temperature,
            context_window=context_window,
            stop_token=stop_token,
            system_prompt=system_prompt,
            is_tool_use_model=is_tool_use_model,
            **generation_kwargs,
        )

    def _call_function(self, tool_call: Tool, response_tool_call: Any):
        """Call the tool function and get the final response"""
        # given the tool_call & tool_name get the tool response
        response_args = response_tool_call["function"]["arguments"]
        args = (
            tool_call.tool_schema.model_validate_json(response_args)
            if isinstance(response_args, str)
            else tool_call.tool_schema.model_validate(response_args)
        ).model_dump()
        # pass the args to the tool_output method, load the json string if required
        # NOTE: the args are already validated upto this point
        output = tool_call.tool_output(args)
        # return the tool type like the openai messages format.
        return {
            "tool_call_id": response_tool_call.get("id", None),
            "role": "tool",
            "name": tool_call.tool_name,
            "content": str(output),
            "return_direct": tool_call.return_direct,
            # The model tool call response to use later.
            "tool_response": {
                "role": "assistant",
                "content": None,
                "tool_calls": [response_tool_call],
            },
        }

    def finalize_response(self, input, scratch_pad, intermediate_steps):
        """Finalize the response of the tool call"""
        # call the chat completion method using the chat history
        # add input & tool_response to chat history
        # NOTE: Is this even right? Do I need to pass the whole chat history for the final response?
        # ANSWER: My thinking is no we don't we create a scratchpad using agent history
        input_message = {"role": "user", "content": input}
        if scratch_pad is not None:
            scratch_pad.append(input_message)
        else:
            scratch_pad = [input_message]
        # print(intermediate_steps)
        for action, observation in intermediate_steps:
            if model_tool_reponse := observation.get("tool_response", None):
                scratch_pad.append(model_tool_reponse)
            else:
                raise ValueError(
                    "Didn't get the function calling response by the model"
                )

            scratch_pad.append(
                {
                    "tool_call_id": observation["tool_call_id"],
                    "role": observation["role"],
                    "name": observation["name"],
                    "content": observation["content"],
                }
            )

            # print(tool_message, scratch_pad)

        response = self.llm.chat(
            {
                "role": "assistant",
                "content": "Received tool responses preparing final response\n",
            },
            scratch_pad,
            tools=None,
        )
        return response["content"], scratch_pad

    def run_step(
        self,
        prompt: str,
        chat_history: Optional[Sequence[dict]] = None,
        scratch_pad: Optional[Sequence[dict]] = [],
        use_tools: bool = True,
    ):
        """Run one execution step"""
        # Validate chat history
        if chat_history:
            self._validate_chat_history(chat_history)
        else:
            chat_history = []

        if use_tools:
            response = self.llm.chat(
                prompt, chat_history + scratch_pad, tools=self.tools
            )
        else:
            response = self.llm.chat(prompt, chat_history + scratch_pad, tools=[])

        return response

    def chat(
        self,
        input: str,
        chat_history: Optional[Sequence[dict]] = None,
        max_retries: int = 3,
        verbose=False,
        *args,
        **kwargs,
    ):
        """Chat using input"""
        trial_no = 0
        final_response = None
        scratch_pad = []  # a list of intermediate toolcall responses by the agent
        while trial_no <= max_retries + 1:  # The first try will be excluede
            # Run the execution loop only until we reach max retries
            try:
                response = self.run_step(
                    input, chat_history, scratch_pad, use_tools=True
                )

                if verbose:
                    print(
                        colored(
                            f"[AGENT]: Model Response: {response}", color="grey"
                        )
                    )
                # if the name field of the response is not Empty
                if tool_calls := response.get("tool_calls", []):
                    # call all the tools
                    intermediate_steps = []
                    for tool in tool_calls:
                        # these tools are already validated for their schema
                        tool_name = tool["function"]["name"]
                        if verbose:
                            print(colored(
                                "[AGENT]: Calling Tool: " + tool_name + "()...", color="grey"
                            ))
                        tool_output = self._call_function(
                            self.tool_dict[tool_name], tool
                        )
                        # make (action, observation) pairs
                        intermediate_steps.append(
                            (tool, tool_output)
                        )  # action is the tool_call, observation is the tool_output
                    # this tool output is the agent scratchpad (in Langchain analogy)
                    # either return the response as it is or finalize it before returning
                    # Add the previous response to the chat_history as well
                    final_response, scratch_pad = (
                        self.finalize_response(input, scratch_pad, intermediate_steps)
                        if not tool_output["return_direct"]
                        else (tool_output["content"], scratch_pad)
                    )

                    return final_response

                else:
                    final_response = response["content"]
                    return final_response

            except ValidationError as e:
                # retry the execution
                if verbose:
                    print(
                        colored(
                            f"[AGENT]: Encountered Validation Error, retrying execution: {e}",
                            color="light_red",
                        )
                    )
                trial_no += 1
                continue

            except KeyError as e:
                # Fallback
                # This error means an unspecified tool was called
                # Small models generally have this problem
                # call the run_step method with use_tools=False to answer from model_knowledge.
                if verbose:
                    print(
                        colored(
                            f"[AGENT]: Encountered KeyError: {e}", color="light_red"
                        )
                    )
                    print(
                        colored(
                            "Attempting to Answer from model Knowledge !", color="yellow"
                        )
                    )
                additional_context = (
                    " Answer from your knowledge & provide a disclaimer to the user."
                )
                response = self.run_step(
                    input + additional_context, chat_history, use_tools=False
                )
                final_response = response["content"]
                return final_response

        # if we come out of the loop unreturned it is certain that we have exceeded the max_tries.
        if final_response is None and trial_no >= max_retries:
            logger.error("MaxRetries exceeded for the agent!!")
            raise Exception(
                "MaxRetries exceeded error!: Reached the max retries for the agent"
            )

        return final_response
