from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from skipnote_core.image.captioning.captioner import Captioner



class BlipCaptioner(Captioner):

    def __init__(self, model_name: str) -> None:
        super().__init__()

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)


    def generate_caption(self, image: Image.Image, **kwargs) -> str:
        inputs = self.processor(images=image, return_tensors="pt")

        outputs = self.model.generate(**inputs)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption


if __name__ == "__main__":
    image_path = "/home/nricciardi/Repositories/skipnote/src/skipnote_core/image/first_extracted_frame.jpg"
    captioner = BlipCaptioner("Salesforce/blip-image-captioning-base")
    caption = captioner.generate_caption_from_path(image_path)

    print(caption)