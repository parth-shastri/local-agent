import ollama
import asyncio
# simple chat with system prompt
# response = ollama.chat(
#     model="qwen2.5:3b",
#     messages=[
#         {"role": "system", "content": "You are a Phd level financial advisior."},
#         {
#             "role": "user",
#             "content": "What are the parameters to look for while investing in a stock?",
#         },
#     ],
#     stream=True,
# )

# for chunk in response:
#     print(chunk["message"]["content"], end="", flush=True)

print("\n" + "-" * 50)

# tool use chat
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city",
                    },
                },
                "required": ["city"],
            },
        },
    },
]
# response = ollama.chat(
#     model="qwen2.5:3b",
#     messages=[{"role": "user", "content": "what is the weather today in Satara, MH?"}],
#     tools=tools,
#     stream=True
# )


# for chunk in response:
#     print(chunk["message"]["content"], end="")

async def main():
    async_client = ollama.AsyncClient()
    response = await async_client.chat(
        model="qwen2.5:3b",
        tools=tools,
        messages=[
            {"role": "system", "content": "You are an helpful assistant."},
            {
                "role": "user",
                "content": "What is the weather in Satara, MH?",
            },
        ],
        stream=True,
    )
    async for chunk in response:
        print(chunk["message"]["content"], end="", flush=True)

asyncio.run(main())

# Result:
# <tool_call>
# {"name": "get_current_weather", "arguments": {"city": "Satara, MH"}}
# </tool_call>%
