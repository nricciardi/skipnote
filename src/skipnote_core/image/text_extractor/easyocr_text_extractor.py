from typing import List
import easyocr
from PIL import Image
import numpy as np
from skipnote_core.image.text_extractor.text_extractor import TextExtractor


class EasyOCRTextExtractor(TextExtractor):

    def __init__(self, languages: List[str], gpu: bool = False) -> None:
        super().__init__()
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def extract_text(self, image: Image.Image, confidence_threshold: float, join_text: str = " ", **kwargs) -> str:
        results = self.reader.readtext(np.array(image))
        text = ""

        for (_, detected_text, confidence) in results:
            if confidence >= confidence_threshold:
                text += detected_text + join_text

        return text.strip()



if __name__ == "__main__":
    import os
    
    path = os.path.join(os.getenv("PYTHONPATH"), "skipnote_core/image/first_extracted_frame.jpg")

    text_extractor = EasyOCRTextExtractor(languages=["en", "it"])
    extracted_text = text_extractor.extract_text_from_path(path, confidence_threshold=0.8, join_text=" ")

    print(extracted_text)