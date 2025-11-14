import numpy as np
from typing import Generator, List
from dataclasses import dataclass, field
from PIL import Image
from skipnote_core.audio.transcriber import AudioTranscriber
from skipnote_core.multimodal.aggregator.block_aggregator import BlockAggregator, InputBlock
from skipnote_core.video.frame_extractor import VideoFrameExtractor
from skipnote_core.text.summarizer.summarizer import TextSummarizer
from skipnote_flow.flow import Flow
import logging



@dataclass
class Block:
    start_time: float
    end_time: float
    text: str
    images: List[Image.Image] = field(default_factory=list)


@dataclass
class SlideBasedVideoFlow(Flow):

    frame_extractor: VideoFrameExtractor
    transcriber: AudioTranscriber
    block_aggregator: BlockAggregator
    frame_difference_threshold: float = 300.0
    join_text_chunks: str = "\n"

    def _compute_frame_difference(self, frame1, frame2) -> float:
        return np.mean((frame1.rgb.astype("float") - frame2.rgb.astype("float")) ** 2)

    def _compute_frame_changes(self, frames: List) -> List[int]:

        frame_differences = [
            self._compute_frame_difference(frames[i], frames[i + 1])
            for i in range(len(frames) - 1)
        ]

        significant_frame_changes_indices = [i + 1 for i, diff in enumerate(frame_differences) if diff >= self.frame_difference_threshold]

        return significant_frame_changes_indices

    def _compute_blocks(self, video_path: str, language: str, mean_slide_duration: float | int) -> Generator[Block, None, None]:
        logging.info(f"Starting computation on video: {video_path}")

        logging.info("Extracting frames...")
        frames = self.frame_extractor.extract_frames(video_path, frame_interval=int(mean_slide_duration))
        logging.info(f"Extracted {len(frames)} frames from video.")

        logging.info("Computing significant frame changes...")
        significant_frame_changes_indices = self._compute_frame_changes(frames)

        logging.info("Transcribing audio...")
        transcription = self.transcriber.transcribe(video_path, language=language)
        logging.info(f"Extracted {len(transcription.chunks)} text chunks from video.")

        current_block_start_time = 0.0
        current_block_images: list[Image.Image] = []
        current_block_texts: list[str] = []

        for chunk in transcription.chunks:
            while significant_frame_changes_indices and chunk.start_time >= frames[significant_frame_changes_indices[0]].timestamp:
                change_index = significant_frame_changes_indices.pop(0)
                current_block_images.append(frames[change_index].to_pillow())

                if current_block_texts:
                    yield Block(
                        start_time=current_block_start_time,
                        end_time=frames[change_index].timestamp,
                        text=self.join_text_chunks.join(current_block_texts),
                        images=current_block_images.copy()
                    )

                    # Reset
                    current_block_start_time = frames[change_index].timestamp
                    current_block_texts.clear()
                    current_block_images.clear()

            current_block_texts.append(chunk.text)

        if current_block_texts:
            yield Block(
                start_time=current_block_start_time,
                end_time=frames[-1].timestamp,
                text=self.join_text_chunks.join(current_block_texts),
                images=current_block_images.copy()
            )

    def run(self, video_path: str, language: str, mean_slide_duration: float | int):

        logging.info("Running...")
        logging.info("Computing blocks...")
        blocks = self._compute_blocks(video_path, language, mean_slide_duration)

        logging.info("Aggregating blocks...")
        output_blocks, summary = self.block_aggregator.aggregate(list(map(
            lambda block: InputBlock(
                text=block.text,
                images=block.images
            ),
            blocks
        )))

        return output_blocks, summary





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

    flow = SlideBasedVideoFlow(
        block_aggregator=BlockAggregator(
            main_generator=OllamaTextGenerator(
                model_name="gemma3:12b"
            ),
            text_filter=OllamaFilter(
                model_name="gemma3:12b",
                language="en"
            ),
            text_summarizer=OllamaSummarizer(
                model_name="gemma3:12b",
                language="en"
            ),
            text_extractor=EasyOCRTextExtractor(languages=["en", "it"], gpu=False)
        ),
        frame_extractor=CV2FrameExtractor(),
        transcriber=FasterWhisperAudioTranscriber("small", device="cuda", compute_type="int8_float16")
    )


    path = os.path.join(os.getenv("PYTHONPATH"), "skipnote_flow/video.mp4")
    
    output_blocks, summary = flow.run(path, language="it", mean_slide_duration=5.0)

    print("Summary:", summary)
    