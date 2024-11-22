from src.models.ollama_model import OllamaModel
from src.models.groq_model import GroqModel
from src.models.llamacpp_model import LlamaCPPModel

MODEL_SERVICES = {
    "ollama": OllamaModel,
    "groq": GroqModel,
    "llamacpp": LlamaCPPModel
}
