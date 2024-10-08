import ollama

# simple chat with system prompt
response = ollama.chat(
    model="llama3.1:latest",
    messages=[
        {"role": "system", "content": "You are a Phd level financial advisior."},
        {
            "role": "user",
            "content": "What are the parameters to look for while investing in a stock?",
        },
    ],
    stream=True,
)

for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)

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
response = ollama.chat(
    model="llama3.1:latest",
    messages=[{"role": "user", "content": "what is the weather today in Satara, MH?"}],
    tools=tools,
)


print("Response for TOOL CALL:")
print(response)
