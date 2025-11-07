from abc import ABC, abstractmethod


class TextSummarizer(ABC):

    @abstractmethod
    def summarize(self, text: str, max_length: int, **kwargs) -> str:
        return NotImplemented