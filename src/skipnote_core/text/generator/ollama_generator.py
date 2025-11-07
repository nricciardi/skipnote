from ollama import generate
from skipnote_core.text.generator.generator import TextGenerator


class OllamaTextGenerator(TextGenerator):

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


    def generate(self, prompt: str) -> str:
        response = generate(
            model=self.model_name,
            prompt=prompt,
        )

        return response['response']

