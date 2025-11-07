from abc import ABC, abstractmethod


class TextFilter(ABC):

    @abstractmethod
    def filter(self, text: str, **kwargs) -> str:
        return NotImplemented