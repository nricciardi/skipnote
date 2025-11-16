from typing import Optional, override
from skipnote_core.text.summarizer.summarizer import TextSummarizer
from skipnote_core.text.generator.ollama_generator import OllamaTextGenerator


class OllamaSummarizer(OllamaTextGenerator, TextSummarizer):

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name=model_name)

        
    @override
    def summarize(self, text: str, max_length: int, language: Optional[str] = None, prompt_template: Optional[str] = None, **kwargs) -> str:

        if prompt_template is None:
            match language:
                case "it":
                    prompt_template = "Riassumi il seguente testo usando al massimo {max_length} parole:\n\n{text}\n\nNon includere altro testo e non usare virgolette. Riassunto:"
                case _:
                    prompt_template = "Summarize the following text using at most {max_length} words:\n\n{text}\n\nDo not append other text and do not use quotes. Summary:"

        prompt_filled = prompt_template.format(text=text, max_length=max_length)
        
        return self.generate(prompt_filled)
    


if __name__ == "__main__":
    text = "La voliera dello zoo era piena di uccelli esotici dai colori sgargianti. Con la nuova macchina fotografica che i nonni mi avevano regalato per il mio compleanno, ho scattato tantissime foto. Dopo aver visitato la voliera, ho detto alla mamma che mi ero divertita un mondo e che non vedevo l'ora di tornare allo zoo."

    summarizer = OllamaSummarizer(model_name="llama3.2:3b")
    summary = summarizer.summarize(text, max_length=10, language="it")
    print(text)
    print(summary)