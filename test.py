from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

llm = Llama.from_pretrained(
    repo_id="meetkai/functionary-small-llama-3.1-GGUF",
    filename="functionary-small-llama-3.1.Q4_0.gguf",
    tokenizer=LlamaHFTokenizer.from_pretrained(
        "meetkai/functionary-small-llama-3.1-GGUF"
    ),
    n_gpu_layers=-1,
    verbose=True,
)
