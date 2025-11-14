import logging
from typing import List, Optional
from dataclasses import dataclass, field
from PIL import Image

from skipnote_core.image.text_extractor.text_extractor import TextExtractor
from skipnote_core.text.filter.filter import TextFilter
from skipnote_core.text.summarizer.summarizer import TextSummarizer
from skipnote_core.text.generator.generator import TextGenerator


@dataclass(frozen=True)
class InputBlock:
    text: str
    images: List[Image.Image] = field(default_factory=list)

@dataclass(frozen=True)
class OutputBlock:
    original_text: str
    processed_text: str
    original_images: List[Image.Image] = field(default_factory=list)
    image_text: List[str] = field(default_factory=list)


@dataclass
class BlockAggregator:

    main_generator: TextGenerator
    text_filter: TextFilter
    text_summarizer: TextSummarizer
    text_extractor: TextExtractor
    text_extraction_confidence_threshold: float = 0.8
    summary_max_length: int = 1000
    context_prompt: str = "Context: You are an expert at fixing transcribed text. Your task is to clean up the text by removing any noise, irrelevant content, and fixing any transcription errors. Provide a clear version of the text without removing any important information.\n\n"
    summary_template: str = "Summary of previous content: {summary}\n\n"
    text_template: str = "Text on which to operate: {text}\n\n"
    image_texts_template: str = "Extracted text from related images: {image_texts}\n\n"
    action_prompt: str = "Result:"


    def _extract_image_text(self, image: Image.Image) -> str:
        image_text = self.text_extractor.extract_text(image, self.text_extraction_confidence_threshold)
        filtered_image_text = self.text_filter.filter(image_text)
        return filtered_image_text
    
    def _update_summary(self, current_summary: Optional[str], new_text: str) -> str:
        if current_summary is None:
            combined_text = new_text
        else:
            combined_text = current_summary + "\n" + new_text
        
        updated_summary = self.text_summarizer.summarize(combined_text, max_length=self.summary_max_length)
        return updated_summary

    def aggregate(self, blocks: List[InputBlock]) -> tuple[List[OutputBlock], Optional[str]]:

        if len(blocks) == 0:
            return [], None

        output_blocks: List[OutputBlock] = []
        summary: Optional[str] = None

        for index, block in enumerate(blocks):

            logging.debug(f"Processing block {index + 1}/{len(blocks)}")

            image_texts = []
            for image in block.images:
                logging.debug(f"Extracting text from image {len(image_texts) + 1}/{len(block.images)} of block {index + 1}")
                image_text = self._extract_image_text(image)
                image_texts.append(image_text)

            logging.debug(f"Generating prompt for block...")
            prompt = self.context_prompt

            if summary is not None:
                prompt += self.summary_template.format(
                    summary=summary
                )
            
            if len(image_texts) > 0:
                prompt += self.image_texts_template.format(
                    image_texts="\n".join(image_texts)
                )

            prompt += self.text_template.format(
                text=block.text
            )
            
            prompt += self.action_prompt

            logging.debug(f"Generating processed text for block...")
            processed_text = self.main_generator.generate(prompt)
            logging.debug(f"Generated processed text length: {len(processed_text)} characters.")

            logging.debug(f"Updating summary with processed text of block...")
            summary = self._update_summary(summary, processed_text)
            logging.debug(f"Updated summary length: {len(summary)} characters.")

            output_block = OutputBlock(
                original_text=block.text,
                processed_text=processed_text,
                original_images=block.images,
                image_text=image_texts
            )
            output_blocks.append(output_block)

        return output_blocks, summary


            
        