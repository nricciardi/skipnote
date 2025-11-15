from abc import ABC, abstractmethod
from typing import Optional
import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal
from .transcription import Transcription


class AudioTranscriber(ABC):

    @classmethod
    def amplify_and_filter_voice(cls, audio_path: str) -> np.ndarray:
        # Load the audio file as a mono 16 kHz time series
        voice_waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

        # Normalize amplitude so the peak absolute value is ≈1.0 (avoid clipping)
        cleaned_waveform = voice_waveform / (np.max(np.abs(voice_waveform)) + 1e-9)

        # Apply band-pass filter to emphasise human voice frequencies + delta (250-3450 Hz)
        sos = signal.butter(8, [250, 3450], btype='bandpass', fs=sample_rate, output='sos')
        filtered_waveform = signal.sosfilt(sos, cleaned_waveform)

        # Apply a small gain to lift low-volume voice somewhat
        gain_factor = 1.5
        amplified_waveform = np.clip(filtered_waveform * gain_factor, -1.0, 1.0)

        # Return the processed waveform and the sample rate
        return amplified_waveform.astype(np.float32)

    @abstractmethod
    def transcribe(self, audio_path: str, language: str, preprocessing: bool = True, word_timestamps: bool = False, initial_prompt: Optional[str] = None, **kwargs) -> Transcription:
        return NotImplemented



