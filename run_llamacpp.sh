# RUN the agent using llamacpp at the backend
python main.py --model_name "meetkai/functionary-small-v2.4-GGUF:functionary-small-v2.4.Q4_0.gguf" --chat_format "functionary-v2" --model_service llamacpp --verbose

# MISTRAL
# python main.py --model_name "meetkai/functionary-small-v2.4-GGUF:functionary-small-v2.4.Q4_0.gguf" --chat_format "functionary-v2" --model_service llamacpp --verbose
