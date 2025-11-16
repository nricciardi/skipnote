from typing import Optional, override
from abc import ABC
from faster_whisper import BatchedInferencePipeline, WhisperModel

from skipnote_core.audio.transcriber import AudioTranscriber
from skipnote_core.audio.transcription import Transcription, TranscriptionChunk, TranscriptionSegment



class FasterWhisperAudioTranscriber(AudioTranscriber):

    def __init__(self, model_name: str, device: str, compute_type: str, batched: bool = False) -> None:
        super().__init__()

        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

        if batched:
            self._model = BatchedInferencePipeline(
                model=self._model
            )

    def _build_transcription_from_segments(self, segments) -> Transcription:
        transcription = Transcription()
        for segment in segments:
            words = None
            if segment.words is not None:
                words = [
                    TranscriptionSegment(
                        start_time=word.start,
                        end_time=word.end,
                        text=word.word,
                        probability=word.probability
                    )
                    for word in segment.words
                ]

            transcription.add_chunk(TranscriptionChunk(
                start_time=segment.start,
                end_time=segment.end,
                text=segment.text.strip(),
                probability=segment.no_speech_prob,
                words=words
            ))

        return transcription

    @override
    def transcribe(self, audio_path: str, language: str, preprocessing: bool = True, word_timestamps: bool = False, initial_prompt: Optional[str] = None,
                   *, temperature: float = 0.0, beam_size: int = 1, condition_on_previous_text: bool = False, batch_size: Optional[int] = None,
                   **kwargs) -> Transcription:

        input = audio_path

        if preprocessing:
            input = self.amplify_and_filter_voice(audio_path)

        other_params = {}
        if isinstance(self._model, BatchedInferencePipeline):
            other_params["batch_size"] = batch_size or 1
        
        segments, info = self._model.transcribe(
            input,
            language=language,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
            vad_filter=True,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            **other_params
        )
        
        return self._build_transcription_from_segments(segments)


if __name__ == "__main__":
    import os

    # path = os.path.join(os.getenv("PYTHONPATH"), "skipnote_core/audio/audio.mp3")
    path = os.path.join(os.getenv("PYTHONPATH"), "skipnote_flow/video.mp4")


    transcriber = FasterWhisperAudioTranscriber("small", device="cuda", compute_type="int8_float16", batched=True)
    transcription = transcriber.transcribe(path, language="en", beam_size=4)

    print(transcription.build_full_text())

    # transcriber = FasterWhisperAudioTranscriber("small", device="cuda", compute_type="int8_float16")
    # transcription = transcriber.transcribe(path, language="en", beam_size=4, word_timestamps=True)

    # print(transcription.build_full_text())