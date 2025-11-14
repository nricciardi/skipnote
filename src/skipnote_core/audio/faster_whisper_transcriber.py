from typing import override

from faster_whisper import BatchedInferencePipeline, WhisperModel

from skipnote_core.audio.transcriber import AudioTranscriber
from skipnote_core.audio.transcription import Transcription, TranscriptionChunk


class FasterWhisperAudioTranscriber(AudioTranscriber):

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        super().__init__()

        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    @override
    def transcribe(self, audio_path: str, language: str, preprocessing: bool = True, *, temperature: float = 0.0, beam_size: int = 1, **kwargs) -> Transcription:

        input = audio_path

        if preprocessing:
            input = self.amplify_and_filter_voice(audio_path)

        segments, info = self._model.transcribe(
            input,
            language=language,
            temperature=temperature,
            condition_on_previous_text=True,
            vad_filter=True,
            beam_size=beam_size,
        )
        
        transcription = Transcription()
        for segment in segments:
            transcription.add_chunk(TranscriptionChunk(
                start_time=segment.start,
                end_time=segment.end,
                text=segment.text
            ))

        return transcription


class FasterWhisperAudioBatchedTranscriber(AudioTranscriber):

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        super().__init__()

        base_model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self._model = BatchedInferencePipeline(
            model=base_model
        )

    @override
    def transcribe(self, audio_path: str, language: str, preprocessing: bool = True, *, temperature: float = 0.0, batch_size: int = 1, beam_size: int = 1, **kwargs) -> Transcription:

        input = audio_path

        if preprocessing:
            input = self.amplify_and_filter_voice(audio_path)

        segments, info = self._model.transcribe(
            input,
            language=language,
            temperature=temperature,
            batch_size=batch_size,
            condition_on_previous_text=True,
            vad_filter=True,
            beam_size=beam_size,
        )
        
        transcription = Transcription()
        for segment in segments:
            transcription.add_chunk(TranscriptionChunk(
                start_time=segment.start,
                end_time=segment.end,
                text=segment.text
            ))

        return transcription


if __name__ == "__main__":
    import os

    # path = os.path.join(os.getenv("PYTHONPATH"), "skipnote_core/audio/audio.mp3")
    path = os.path.join(os.getenv("PYTHONPATH"), "skipnote_flow/video.mp4")
    transcriber = FasterWhisperAudioBatchedTranscriber("small", device="cuda", compute_type="int8_float16")
    transcription = transcriber.transcribe(path, language="it", batch_size=4, beam_size=4)

    print(transcription.build_full_text())

    transcriber = FasterWhisperAudioTranscriber("small", device="cuda", compute_type="int8_float16")
    transcription = transcriber.transcribe(path, language="it", beam_size=4)

    print(transcription.build_full_text())