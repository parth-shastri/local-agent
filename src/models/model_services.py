from src.models.ollama_model import OllamaModel
from src.models.groq_model import Groq
from src.models.llamacpp_model import LlamaCPPModel

MODEL_SERVICES = {
    "ollama": OllamaModel,
    "groq": Groq,
    "llamacpp": LlamaCPPModel
}
