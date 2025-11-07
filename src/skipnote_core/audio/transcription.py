from dataclasses import dataclass, field
from typing import List


@dataclass
class TranscriptionChunk:
    start_time: float
    end_time: float
    text: str

@dataclass
class Transcription:
    chunks: List[TranscriptionChunk] = field(default_factory=list)

    def add_chunk(self, chunk: TranscriptionChunk):
        self.chunks.append(chunk)

    def build_full_text(self) -> str:
        return " ".join(chunk.text for chunk in self.chunks)
