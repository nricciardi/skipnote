from abc import ABC, abstractmethod
from typing import Optional


class TextSummarizer(ABC):

    @abstractmethod
    def summarize(self, text: str, max_length: int, language: Optional[str] = None, **kwargs) -> str:
        return NotImplemented