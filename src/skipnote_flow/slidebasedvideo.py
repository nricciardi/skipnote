import datetime
import uuid
import os
import json
import numpy as np
from typing import Dict, Generator, List, Optional
from dataclasses import dataclass, field
from PIL import Image
from skipnote_core.audio.transcriber import AudioTranscriber
from skipnote_core.audio.transcription import Transcription
from skipnote_core.multimodal.aggregator.block_aggregator import BlockAggregator, InputBlock
from skipnote_core.video.frame_extractor import VideoFrameExtractor
from skipnote_core.text.summarizer.summarizer import TextSummarizer
from skipnote_flow.flow import Flow
import logging



@dataclass
class Section:
    start_time: float
    end_time: float
    text: Optional[str]
    images: List[Image.Image] = field(default_factory=list, kw_only=True)

    def to_dict(self, dir_path: str, *, replace_images_with_paths: bool = True, image_dir: str = "images") -> Dict:

        image_dir_path = os.path.join(dir_path, image_dir)
        os.makedirs(image_dir_path, exist_ok=True)

        if replace_images_with_paths:
            image_paths = []
            for image in self.images:
                image_path = os.path.join(image_dir_path, f"{uuid.uuid4().hex}.png")
                image.save(image_path)
                image_paths.append(image_path)

            return {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "text": self.text,
                "images": image_paths,
            }
        
        else:
            return {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "text": self.text,
                "images": self.images,
            }


@dataclass
class ProcessedSection(Section):
    processed_text: str
    image_texts: List[str]

    def to_dict(self, dir_path: str, **kwargs) -> Dict:
        base_data = super().to_dict(dir_path, **kwargs)
        base_data.update({
            "processed_text": self.processed_text,
            "image_texts": self.image_texts,
        })
        return base_data


@dataclass
class Outcome:
    created_at: datetime.datetime
    elaboration_time: float
    sections: List[Section]
    summary: Optional[str] = None

    def save(self, dir_path: str) -> None:
        
        os.makedirs(dir_path, exist_ok=True)

        with open(os.path.join(dir_path, "outcome.json"), "w", encoding="utf-8") as f:
            json.dump({
                "created_at": self.created_at.isoformat(),
                "elaboration_time": self.elaboration_time,
                "summary": self.summary,
                "sections": [
                    section.to_dict(dir_path, replace_images_with_paths=True)
                    for section in self.sections
                ],
            }, f, indent=4)


@dataclass
class SlideBasedVideoFlow(Flow):

    frame_extractor: VideoFrameExtractor
    transcriber: AudioTranscriber
    block_aggregator: BlockAggregator
    frame_difference_threshold: float = 300.0

    def _compute_frame_difference(self, frame1, frame2) -> float:
        return np.mean((frame1.rgb.astype("float") - frame2.rgb.astype("float")) ** 2)

    def _compute_frame_changes(self, frames: List) -> List[int]:

        frame_differences = [
            self._compute_frame_difference(frames[i], frames[i + 1])
            for i in range(len(frames) - 1)
        ]

        significant_frame_changes_indices = [i + 1 for i, diff in enumerate(frame_differences) if diff >= self.frame_difference_threshold]

        return significant_frame_changes_indices
    
    def compute_sections(self, video_path: str, language: str, *, min_slide_duration: float, setup_time: Optional[float], explicit_word_timestamps: bool = False, **kwargs) -> Generator[Section, None, None]:
        logging.info(f"Starting computation on video: {video_path}")

        logging.info("Extracting frames...")
        frames = self.frame_extractor.extract_frames(video_path, frame_interval=int(min_slide_duration))
        logging.info(f"Extracted {len(frames)} frames from video.")

        logging.info("Computing significant frame changes...")
        relevant_frame_changes_indices = self._compute_frame_changes(frames)

        logging.info("Transcribing audio...")
        transcription = self.transcriber.transcribe(video_path, language=language, word_timestamps=explicit_word_timestamps, **kwargs)
        logging.info(f"Extracted {len(transcription.chunks)} text chunks from video.")

        # Filter out frame changes that occur before setup_time
        if setup_time is not None:
            relevant_frame_changes_indices = [
                index for index in relevant_frame_changes_indices
                if frames[index].timestamp >= setup_time
            ]

            words = transcription.get_words_in_interval(0.0, setup_time)
            yield Section(
                start_time=0.0,
                end_time=setup_time,
                text=" ".join(words) if words else None,
                images=[],
            )
            

        for i in range(1, len(relevant_frame_changes_indices)):
            previous_relevant_frame_index = relevant_frame_changes_indices[i - 1]
            current_relevant_frame_index = relevant_frame_changes_indices[i]

            start_time = frames[previous_relevant_frame_index].timestamp
            end_time = frames[current_relevant_frame_index].timestamp

            words = transcription.get_words_in_interval(start_time, end_time)

            yield Section(
                start_time=start_time,
                end_time=end_time,
                text=" ".join(words) if words else None,
                images=[frames[previous_relevant_frame_index].to_pil_image()],
            )

        last_frame = frames[relevant_frame_changes_indices[-1]]

        words = transcription.get_words_in_interval(last_frame.timestamp, transcription.end_time)
        yield Section(
            start_time=last_frame.timestamp,
            end_time=transcription.end_time,
            text=" ".join(words) if words else None,
            images=[last_frame.to_pil_image()],
        )

    def run(self, video_path: str, language: str, post_processing: bool, min_slide_duration: float, setup_time: Optional[float] = None, **kwargs) -> Outcome:

        init_time = datetime.datetime.now()

        logging.info("Running...")

        logging.info("Computing sections...")
        sections = list(self.compute_sections(video_path, language, min_slide_duration=min_slide_duration, setup_time=setup_time, **kwargs))

        logging.info(f"Computed {len(sections)} sections in {(datetime.datetime.now() - init_time).total_seconds()} seconds.")

        if not post_processing:
            logging.info("Skipping post-processing as per configuration.")
            
            return Outcome(
                created_at=datetime.datetime.now(),
                elaboration_time=(datetime.datetime.now() - init_time).total_seconds(),
                sections=sections
            )

        logging.info("Aggregating sections...")

        input_blocks: List[InputBlock] = []
        for section in sections:
            if section.text is None and len(section.images) == 0:
                logging.warning(f"Skipping empty section from {section.start_time} to {section.end_time}")
                continue

            if section.text is not None:
                logging.warning(f"Section text length: {len(section.text)} characters.")

            if len(section.images) > 0:
                logging.warning(f"Section contains {len(section.images)} images.")

            input_blocks.append(InputBlock(
                text=section.text if section.text is not None else "",
                images=section.images
            ))

        output_blocks, summary = self.block_aggregator.aggregate(input_blocks)

        outcome_sections: List[ProcessedSection] = []
        for section, output_block in zip(sections, output_blocks):
            outcome_sections.append(ProcessedSection(
                start_time=section.start_time,
                end_time=section.end_time,
                text=section.text,
                images=section.images,
                processed_text=output_block.processed_text,
                image_texts=output_block.image_text
            ))
            

        return Outcome(
            created_at=datetime.datetime.now(),
            elaboration_time=(datetime.datetime.now() - init_time).total_seconds(),
            sections=outcome_sections,
            summary=summary
        )
            



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    import os
    from skipnote_core.video.cv2_frame_extractor import CV2FrameExtractor
    from skipnote_core.audio.faster_whisper_transcriber import FasterWhisperAudioTranscriber
    from skipnote_core.multimodal.aggregator.block_aggregator import BlockAggregator
    from skipnote_core.text.summarizer.ollama_summarizer import OllamaSummarizer
    from skipnote_core.text.filter.ollama_filter import OllamaFilter
    from skipnote_core.text.generator.ollama_generator import OllamaTextGenerator
    from skipnote_core.image.text_extractor.easyocr_text_extractor import EasyOCRTextExtractor

    ollama_model = "llama3.2:3b"

    flow = SlideBasedVideoFlow(
        block_aggregator=BlockAggregator(
            main_generator=OllamaTextGenerator(
                model_name=ollama_model
            ),
            text_filter=OllamaFilter(
                model_name=ollama_model,
                language="en"
            ),
            text_summarizer=OllamaSummarizer(
                model_name=ollama_model,
                language="en"
            ),
            text_extractor=EasyOCRTextExtractor(languages=["en"], gpu=False)
        ),
        frame_extractor=CV2FrameExtractor(),
        transcriber=FasterWhisperAudioTranscriber("small", device="cuda", compute_type="int8_float16")
    )


    input_path = os.path.join(os.getenv("PYTHONPATH"), "skipnote_flow/video.mp4")
    output_path = os.path.join(os.getenv("PYTHONPATH"), "skipnote_flow/outcome")
    
    outcome = flow.run(input_path, language="en", post_processing=False, min_slide_duration=5.0, beam_size=4)

    print("Summary:", outcome.summary)

    outcome.save(output_path)
    