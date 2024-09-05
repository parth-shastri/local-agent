# A class to query the ollama service running on the localhost
import requests
import json


class OllamaModel:
    def __init__(self, model, system_prompt, temperature=0, stop=None):
        """
        Init the OllamaModel with the given parameters

        Parameters:
            model (str): The name of the model to use.
            system_prompt (str): The system prompt to use.
            temperature (float): The temperature setting for the model.
            stop (str): The stop token for the model.
        """
        self.model_endpoint = "http://localhost:11434/api/generate"
        self.temperature = temperature
        self.model = model
        self.system_prompt = system_prompt
        self.stop = stop
        self.headers = {"Content-Type": "application/json"}

    def generate_text(self, prompt):
        """
        Generates response from the Ollama model, based on the provided prompt.
        """
        payload = {
            "model": self.model,
            "format": 'json',
            "prompt": prompt,
            "system_prompt": self.system_prompt,
            "stream": False,
            "temperature": self.temperature,
            "stop": self.stop
        }

        try:
            request_response = requests.post(
                self.model_endpoint,
                headers=self.headers,
                data=json.dumps(payload),
            )
            print(f"REQUEST RESPONSE: {request_response}")
            request_response_json = request_response.json()
            response = request_response_json['response']
            response_dict = json.loads(response)

            print(f"\n\nResponse from OllamaModel::{self.model}={response_dict}")

            return response_dict

        except requests.RequestException as e:
            response = {"error": f"Error in invoking the model: {str(e)}"}

            return response
