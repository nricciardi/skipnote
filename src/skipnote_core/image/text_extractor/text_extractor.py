from abc import ABC, abstractmethod
from PIL import Image


class TextExtractor(ABC):
    @abstractmethod
    def extract_text(self, image: Image.Image, confidence_threshold: float, join_text: str = " ", **kwargs) -> str:
        return NotImplemented
    
    def extract_text_from_path(self, image_path: str, confidence_threshold: float, join_text: str = " ", **kwargs) -> str:
        image = Image.open(image_path)
        return self.extract_text(image, confidence_threshold, join_text, **kwargs)