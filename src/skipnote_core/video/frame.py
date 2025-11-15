from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image


@dataclass
class Frame(ABC):
    timestamp: float

    @property
    @abstractmethod
    def rgb(self) -> np.ndarray:
        return NotImplemented
    
    def to_pil_image(self) -> Image.Image:
        return Image.fromarray(self.rgb)
    
    def save(self, file_path: str) -> None:
        img = self.to_pil_image()
        img.save(file_path)


@dataclass
class InMemoryFrame(Frame):
    data: np.ndarray

    @property
    def rgb(self) -> np.ndarray:
        return self.data

@dataclass
class OnDiskFrame(Frame):
    file_path: str

    @property
    def rgb(self) -> np.ndarray:
        return NotImplemented
