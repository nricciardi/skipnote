from abc import ABC, abstractmethod
from typing import Optional


class TextFilter(ABC):

    @abstractmethod
    def filter(self, text: str, language: Optional[str] = None, **kwargs) -> str:
        return NotImplemented