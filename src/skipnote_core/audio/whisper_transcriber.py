from typing import Optional, override

import whisper

from skipnote_core.audio.transcriber import AudioTranscriber
from skipnote_core.audio.transcription import Transcription, TranscriptionChunk


class WhisperAudioTranscriber(AudioTranscriber):

    def __init__(self, model_name: str, device: str) -> None:
        super().__init__()

        self._model = whisper.load_model(model_name, device=device)

    @override
    def transcribe(self, audio_path: str, language: str, preprocessing: bool = True, *, temperature: float = 0.0, beam_size: int = 1, initial_prompt: Optional[str] = None, **kwargs) -> Transcription:

        return NotImplemented
