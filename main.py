# The main event loop to listen to user messages
import os
import logging
import time
from logging_utils.init_logging import init_logging
from src.agents.function_calling_agent import FunctionCallingAgent
from src.tools import model_tools
from src.tools.base import Tool
from src.utils import calculate_token_count_of_message, truncate_chat_history
from argparse import ArgumentParser
from termcolor import colored
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import pyfiglet
from configs.config import GROQ_API_KEY

# init logging
init_logging("./configs/logging.json")
logger = logging.getLogger(__name__)

# init the console
console = Console()


def print_ascii_art_for_local_agent():
    """Generate ascii art for 'local-agent' using pyfiglet"""
    art = pyfiglet.figlet_format(text="local - agent", font="slant")
    console.print(f"[bold cyan]{art.center(100)}[/bold cyan]")


def clear():
    """Clear the terminal screen"""
    print("\033c", end='')


def get_user_input():
    """
    Get the user's input from the command line with a styled prompt.

    Returns:
        str: The user's input.
    """
    return Prompt.ask("[bold cyan]Ask me anything:[/bold cyan]")


def init_agent_and_tools(args):
    # tools to use
    # NOTE: define any custom tools if want to use under ./src/tools/model_tools.py
    logger.info(f"Tools found {model_tools.__all__}")

    all_tools = model_tools.__all__

    # make a tool list
    tools = [
        Tool.from_function(function=getattr(model_tools, tool), return_direct=False)
        for tool in all_tools
    ]

    generation_kwargs = {"max_tokens": args.max_tokens}

    agent = FunctionCallingAgent(
        model_name=args.model_name,
        model_service=args.model_service,
        chat_format=args.chat_format,
        tools=tools,
        stop_token=args.stop_token,
        system_prompt=args.system_prompt,
        is_tool_use_model=args.is_tool_use,
        generation_kwargs=generation_kwargs,
        model_verbose=True,
    )

    # Main Listening loop
    logger.info(
        colored(
            f"Session with agent based on {args.model_name} initialized.",
            color="magenta",
        )
    )
    logger.info(f"Verbose is set to {args.verbose}")
    return agent


def get_args():
    """
    Command line argument parser.
    """
    parser = ArgumentParser(description="A Local Agent that has access to tools.")
    parser.add_argument("--model_name", '-m', type=str, default="llama3.1:latest", help="The model to use, NOTE: Download the model first using `ollama run model_name` if using ollama models.")
    parser.add_argument("--model_service", default="ollama", choices=["ollama", "llamacpp", "groq"], help="The model_service provider that hosts the model.")
    parser.add_argument("--chat_format", type=str, default=None, help="The chat format to use (only for llamacpp models)")
    parser.add_argument("--is_tool_use", action="store_true", default=True, help="If the tool use capabilities of the model are natively supported by the model_service.")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--context_window", type=int, default=4000)
    parser.add_argument("--stop_token", "-st", type=list, help="Tokens that determine the end of sequence, may be different for different models.")
    parser.add_argument("--verbose", action='store_true', help="Displays the agent internal calls for debugging", default=False)
    parser.add_argument("--system_prompt", type=str, help="The system prompt for the model. will use the default prompt under src/prompts/prompts.py", default=None)
    parser.add_argument("--max_tokens", type=int, help="The maximum number of tokens to generate.", default=1024)

    return parser.parse_args()


def main(args):
    # check the API key if groq
    if args.model_service:
        assert os.environ.get("GROQ_API_KEY") == GROQ_API_KEY
        logger.info("API key matched !!")

    agent = init_agent_and_tools(args)

    # print the ascii art
    print_ascii_art_for_local_agent()
    console.print(
        "[yellow]Type your query below, type 'quit' or 'exit' to exit & 'restart' to restart the session.[/yellow]\n")
    chat_memory = None
    while True:
        # get the user input messages
        prompt = get_user_input()
        if prompt.lower() in ["exit", "quit"]:
            console.print("[cyan]Goodbye[/cyan]")
            break
        elif prompt.lower() == "restart":
            clear()
            logger.info(colored("Session restarted", color="yellow"))
            print_ascii_art_for_local_agent()
            console.print(
                "[yellow]Type your query below, type 'quit' or 'exit' to exit & 'restart' to restart the session.[/yellow]\n"
            )
            chat_memory = None

        else:
            # truncate the chat and calculate the tokens
            system_tokens = (
                calculate_token_count_of_message(agent.system_prompt)
                if agent.system_prompt
                else 0
            )
            prompt_tokens = calculate_token_count_of_message(prompt, tokenizer=None)
            if chat_memory is not None:
                chat_memory, chat_memory_tokens = truncate_chat_history(
                    chat_memory, token_limit=1024, tokenizer=None
                )
            else:
                chat_memory_tokens = 0

            if (
                system_tokens + chat_memory_tokens + prompt_tokens
                > agent.context_window - args.max_tokens
            ):
                logger.warning(
                    colored(
                        "Max token counts reached! Please start a new session. Type 'restart' in the console.", color="red"
                    )
                )
                break

            start = time.perf_counter()
            response = agent.chat(
                prompt, chat_history=chat_memory, max_retries=3, verbose=args.verbose
            )
            end = time.perf_counter()
            console.print(f"[green]\nResponse: {response}[/green]")
            # Calcuate the time to response
            time_to_response = end - start
            # Display timing information
            timing_info = f"""
            [cyan]Total Response Time:[/cyan] [bold yellow]{time_to_response:.2f} seconds[/bold yellow]
            """
            console.print(
                Panel(
                    timing_info, title="[bold green]Response Timing[/bold green]", expand=False
                )
            )

            # structure the user and the response
            # add both user and assistant messages to the chat history
            # NOTE: There is no system message added in the chat history
            user_message = {"role": "user", "content": prompt}
            response_message = {"role": "assistant", "content": response}

            if chat_memory is None:
                chat_memory = [user_message, response_message]
            else:
                chat_memory.extend([user_message, response_message])


if __name__ == "__main__":

    # parse the arguments
    args = get_args()

    # run the main function
    main(args)
