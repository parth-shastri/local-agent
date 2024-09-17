# The main event loop to listen to user messages
from src.agents.function_calling_agent import FunctionCallingAgent
from src.tools import model_tools
from src.tools.base import Tool
from src.utils import calculate_token_count_of_message, truncate_chat_history
from argparse import ArgumentParser
from termcolor import colored


def get_args():
    """
    Command line argument parser.
    """
    parser = ArgumentParser(description="A Local Agent that has access to tools.")
    parser.add_argument("--model_path", "-mp", type=str, default="meta-llama/llama3.1-8b", help="The huggingface repo-id/ model_path to use, (only used in case llama_cpp is the model_service)")
    parser.add_argument("--model_name", '-m', type=str, default="llama3.1:latest", help="The model to use, NOTE: Download the model first using `ollama run model_name` if using ollama models.")
    parser.add_argument("--model_service", default="ollama", choices=["ollama", "llamacpp"], help="The model_service provider that hosts the model.")
    parser.add_argument("--is_tool_use", action="store_true", default=True, help="If the tool use capabilities of the model are natively supported by the model_service.")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--context_window", type=int, default=4000)
    parser.add_argument("--stop_token", "-st", type=list, help="Tokens that determine the end of sequence, may be different for different models.")
    parser.add_argument("--verbose", action='store_true', help="Displays the agent internal calls for debugging", default=False)
    parser.add_argument("--system_prompt", type=str, help="The system prompt for the model. will use the default prompt under src/prompts/prompts.py", default=None)

    return parser.parse_args()


if __name__ == "__main__":

    # parse the arguments
    args = get_args()

    # tools to use
    # NOTE: define any custom tools if want to use under ./src/tools/model_tools.py
    print(f"Tools found {model_tools.__all__}")

    all_tools = model_tools.__all__

    # make a tool list
    tools = [Tool.from_function(function=getattr(model_tools, tool)) for tool in all_tools]

    agent = FunctionCallingAgent(
        model_path=args.model_path,
        model_name=args.model_name,
        model_service=args.model_service,
        tools=tools,
        stop_token=args.stop_token,
        system_prompt=args.system_prompt,
        is_tool_use_model=args.is_tool_use
    )

    chat_memory = None
    # Main Listening loop
    print(
        colored(
            f"Session with agent based on {args.model_name} initialized.",
            color="magenta",
        )
    )
    while True:
        # get the user input messages

        prompt = input("Ask me anything: ")
        if prompt.lower() == "exit":
            break

        # truncate the chat and calculate the tokens
        system_tokens = calculate_token_count_of_message(agent.system_prompt)
        prompt_tokens = calculate_token_count_of_message(prompt, tokenizer=None)
        if chat_memory is not None:
            chat_memory, chat_memory_tokens = truncate_chat_history(chat_memory, token_limit=512, tokenizer=None)
        else:
            chat_memory_tokens = 0

        if system_tokens + chat_memory_tokens + prompt_tokens > agent.context_window - 2048:
            print(colored("Max token counts reached! Please start a new session.", color='red'))
            break

        response = agent.chat(prompt, chat_history=chat_memory, max_retries=3, verbose=args.verbose)

        print(response)

        # structure the user and the response
        # add both user and assistant messages to the chat history
        # NOTE: There is no system message added in the chat history
        user_message = {'role': 'user', 'content': prompt}
        response_message = {"role": "assistant", "content": response}

        if chat_memory is None:
            chat_memory = [user_message, response_message]
        else:
            chat_memory.extend([user_message, response_message])
