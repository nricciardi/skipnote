from typing import Optional, override
from skipnote_core.text.filter.filter import TextFilter
from skipnote_core.text.generator.ollama_generator import OllamaTextGenerator


class OllamaFilter(OllamaTextGenerator, TextFilter):

    def __init__(self, model_name: str, context: Optional[str] = None) -> None:
        super().__init__(model_name=model_name)
        
        self.context = context

            
    @override
    def filter(self, text: str, language: Optional[str] = None, **kwargs) -> str:
        match language:
            case "it":
                if self.context is not None:
                    self.prompt_template += f"Contesto: {self.context}\n\n"
                self.prompt_template += "Rimuovi dal seguente testo rumore e contenuti irrilevanti:\n\n{text}\n\nNon includere altro testo e non usare virgolette. Risultato:"
            case _:
                self.prompt_template = ""
                if self.context is not None:
                    self.prompt_template += f"Context: {self.context}\n\n"
                self.prompt_template += "Remove from following text noise and irrelevant content:\n\n{text}\n\nDo not append other text and do not use quotes. Result:"

        prompt_filled = self.prompt_template.format(text=text)
        
        return self.generate(prompt_filled)


if __name__ == "__main__":
    text = "GA GHIGGIA ADA QUALI TEST? ML Test di screening (Frontal Assessment Battery) CS Fluenza alternata Wisconsing Card Test Test per la valutazione della working memory: CM Digit Span e Test di Corsi Matrici di Raven MILITELLO CHL Test per la valutazione della pianificazione, problem OM Torre di Londra OTERI MARTA [ GHIGGIA ADA Sorting"

    filter = OllamaFilter(model_name="gemma3:12b", context="Lezione universitaria di psicologia.")
    filtered_text = filter.filter(text, language="it")
    print(text)

    print("----- FILTERED TEXT -----")
    print(filtered_text)