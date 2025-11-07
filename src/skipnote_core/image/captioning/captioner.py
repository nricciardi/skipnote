from PIL import Image
from abc import ABC, abstractmethod


class Captioner(ABC):
    @abstractmethod
    def generate_caption(self, image: Image.Image, **kwargs) -> str:
        return NotImplemented

    def generate_caption_from_path(self, image_path: str, **kwargs) -> str:
        image = Image.open(image_path)
        return self.generate_caption(image, **kwargs)