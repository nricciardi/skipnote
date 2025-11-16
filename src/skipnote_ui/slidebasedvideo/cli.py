#!/venv/bin/python3

import argparse
import os
import logging
from time import sleep
import torch
from skipnote_core.video.cv2_frame_extractor import CV2FrameExtractor
from skipnote_core.audio.faster_whisper_transcriber import FasterWhisperAudioTranscriber
from skipnote_core.multimodal.aggregator.block_aggregator import BlockAggregator
from skipnote_core.text.summarizer.ollama_summarizer import OllamaSummarizer
from skipnote_core.text.filter.ollama_filter import OllamaFilter
from skipnote_core.text.generator.ollama_generator import OllamaTextGenerator
from skipnote_core.image.text_extractor.easyocr_text_extractor import EasyOCRTextExtractor
from skipnote_flow.slidebasedvideo import SlideBasedVideoFlow


logging.basicConfig(level=logging.INFO)


def get_parser():
    parser = argparse.ArgumentParser(
        description="CLI for video processing with Ollama model"
    )

    parser.add_argument(
        "--ollama-model",
        type=str,
        required=True,
        help="Name of the Ollama model to use"
    )

    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language to use for transcription"
    )

    parser.add_argument(
        "--transcriber-model",
        type=str,
        required=True,
        help="Size of the transcription model"
    )

    parser.add_argument(
        "--transcriber-compute-type",
        type=str,
        required=False,
        default="int8_float16",
        help="Compute type for the transcription model (default: int8_float16)"
    )

    parser.add_argument(
        "--transcriber-beam-size",
        type=int,
        required=False,
        default=1,
        help="Beam size for the transcription model (default: 1)"
    )

    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to the video file to process"
    )

    parser.add_argument(
        "--post-processing",
        action="store_true",
        help="Enable post-processing (default: disabled)"
    )

    parser.add_argument(
        "--min-slide-duration",
        type=float,
        default=5.0,
        help="Minimum slide duration (default: 5)"
    )

    parser.add_argument(
        "--setup-time",
        type=float,
        default=None,
        help="Optional setup time (default: None)"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for the results"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation (default: cuda if available, else cpu)"
    )

    parser.add_argument(
        "--export-markdown",
        action="store_true",
        help="Export output in Markdown format (default: false)"
    )

    return parser

def main():
    logging.info("Starting Ollama...")
    os.popen("ollama serve > /dev/null 2>&1 &")

    sleep(2)  # Give some time for Ollama to start

    logging.info("Available Ollama models:")
    logging.info(f"\n\n{os.popen('ollama list').read()}\n\n")


    parser = get_parser()
    args = parser.parse_args()


    logging.info(f"Ollama model selected for this run: {args.ollama_model}")
    
    flow = SlideBasedVideoFlow(
        block_aggregator=BlockAggregator(
            main_generator=OllamaTextGenerator(
                model_name=args.ollama_model
            ),
            text_filter=OllamaFilter(
                model_name=args.ollama_model,
            ),
            text_summarizer=OllamaSummarizer(
                model_name=args.ollama_model,
            ),
            text_extractor=EasyOCRTextExtractor(languages=[args.language], gpu=False)
        ),
        frame_extractor=CV2FrameExtractor(),
        transcriber=FasterWhisperAudioTranscriber(args.transcriber_model, device=args.device, compute_type=args.transcriber_compute_type)
    )

    outcome = flow.run(args.video_path, language=args.language, post_processing=args.post_processing, min_slide_duration=args.min_slide_duration, beam_size=args.transcriber_beam_size)

    logging.info(f"Saving outcome to {args.output_path}")

    data = outcome.save(args.output_path)

    if args.export_markdown:
        md_path = os.path.join(args.output_path, "outcome.md")
        logging.info(f"Exporting outcome to Markdown at {md_path}")

        md_content = outcome.as_markdown(data, relative_image_base_path=args.output_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
