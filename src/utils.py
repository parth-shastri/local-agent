from typing import Union, Optional
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from collections import deque


def calculate_token_count_of_message(
    message: Union[str, dict],
    tokenizer: Optional[Union[str, PreTrainedTokenizerBase]] = None,
):
    """
    Given a list of messages in openai format, calculate the total token count of the messages.
    """
    if not tokenizer:
        # if a tokenizer is not specified
        avg_char_per_token = 4    # this is the general consensus

        if isinstance(message, dict):
            # this should be in the openai format
            message = message['content']
        # calculate the total chars
        char_count = len(message)
        no_of_tokens_message = char_count // avg_char_per_token  # integer
        return no_of_tokens_message

    elif isinstance(tokenizer, str):
        # use hugging face tokenizer to count tokens
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    encoded_message = tokenizer.encode(message)
    no_of_tokens_message = len(encoded_message)

    return no_of_tokens_message


def truncate_chat_history(
    messages,
    token_limit=1536,
    tokenizer: Optional[Union[str, PreTrainedTokenizerBase]] = None,
):
    """
    Limits the chat history to given token_limit.
    """
    def truncate(message_queue: deque, total_tokens):
        while total_tokens > token_limit:
            _, message_token_count = message_queue.popleft()
            # subract the tokencount
            total_tokens -= message_token_count
            # print(f"Reduced token count: {total_tokens}")
            return truncate(message_queue, total_tokens)
        return message_queue

    message_queue = deque()
    total_token_count = 0
    # calculate the token count of each message
    for message in messages:
        token_count = calculate_token_count_of_message(message, tokenizer=tokenizer)
        message_queue.append((message, token_count))
        total_token_count += token_count

    # print(f"Total token count: {total_token_count}")
    if total_token_count <= token_limit:
        return messages, total_token_count
    else:
        truncated_queue = truncate(message_queue, total_token_count)
        # make messages from the queue

        messages, total_truncated = [], 0
        for message, c in truncated_queue:
            total_truncated += c
            messages.append(message)

        return messages, total_truncated
