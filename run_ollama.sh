## CUDA PATHS ##
# export PATH="/usr/local/cuda-12.2/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"
# export PATH="$PATH:/home/ostrich/.local/bin"

# RUN the agent using ollama
python main.py --model_name "qwen2.5:3b" --model_service ollama --verbose --system_prompt "You are a helpful assistant agent"