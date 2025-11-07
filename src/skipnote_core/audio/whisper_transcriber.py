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

        input = audio_path

        if preprocessing:
            input = self.amplify_and_filter_voice(audio_path)

        result = self._model.transcribe(
            input,
            language=language,
            temperature=temperature,
            condition_on_previous_text=True,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
        )
        
        transcription = Transcription()
        for segment in result["segments"]:
            transcription.add_chunk(TranscriptionChunk(
                start_time=segment["start"],
                end_time=segment["end"],
                text=segment["text"]
            ))

        return transcription


if __name__ == "__main__":
    audio_path = "/home/nricciardi/Repositories/skipnote/src/skipnote_core/audio/audio.mp3"
    transcriber = WhisperAudioTranscriber("medium", device="cuda")
    transcription = transcriber.transcribe(audio_path, language="it", beam_size=4, initial_prompt="Questa è una videolezione di psicologia.")

    print(transcription.build_full_text())