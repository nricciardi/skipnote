from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TranscriptionSegment:
    start_time: float
    end_time: float
    text: str
    probability: float = 0.0


@dataclass
class TranscriptionChunk(TranscriptionSegment):
    words: Optional[List[TranscriptionSegment]] = None

@dataclass
class Transcription:
    chunks: List[TranscriptionChunk] = field(default_factory=list)

    def add_chunk(self, chunk: TranscriptionChunk):
        self.chunks.append(chunk)

    def build_full_text(self) -> str:
        return " ".join(chunk.text for chunk in self.chunks)
    
    @property
    def start_time(self) -> float:
        if not self.chunks:
            return 0.0
        return self.chunks[0].start_time
    
    @property
    def end_time(self) -> float:
        if not self.chunks:
            return 0.0
        return self.chunks[-1].end_time
    
    def get_chunk_words_in_interval(self, from_time: float, to_time: float) -> List[List[str]]:
        chunks = []

        for chunk in self.chunks:
            if chunk.end_time < from_time:
                continue
            if chunk.start_time > to_time:
                continue

            if chunk.words:
                words = []
                for word in chunk.words:
                    if word.start_time >= from_time and word.end_time <= to_time:
                        words.append(word.text)

                chunks.append(words)
            else:
                raise ValueError("Chunk does not contain word-level timestamps.")

        return chunks
    
    def get_words_in_interval(self, from_time: float, to_time: float) -> List[str]:
        words = []

        chunk_words = self.get_chunk_words_in_interval(from_time, to_time)
        for word_list in chunk_words:
            words.extend(word_list)

        return words
    
