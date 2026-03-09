"""Code for Whisper transcription using HuggingFace's transformers library."""
import wave
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from .const import Transcriber

_RATE = 16000

class TransformersTranscriber(Transcriber):
    """Wrapper for HuggingFace transformers Whisper model."""
    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        device: str = "cpu",
    ) -> None:
        """Initialize Whisper model."""
        self.device = device
        torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
        model.to(device)
        self.is_multilingual = getattr(model.generation_config, "is_multilingual", True)
        if device != "cpu":
            import intel_extension_for_pytorch as ipex
            model = ipex.optimize(model, dtype=torch_dtype)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Returns transcription for WAV file.
        WAV file must be 16Khz 16-bit mono audio.
        """
        wav_file: wave.Wave_read = wave.open(str(wav_path), "rb")
        with wav_file:
            assert wav_file.getframerate() == _RATE, "Sample rate must be 16Khz"
            assert wav_file.getsampwidth() == 2, "Width must be 16-bit (2 bytes)"
            assert wav_file.getnchannels() == 1, "Audio must be mono"
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        generate_kwargs: dict = {"num_beams": beam_size}
        if self.is_multilingual:
            generate_kwargs["task"] = "transcribe"
            if language:
                generate_kwargs["language"] = language

        # Handle initial_prompt by converting it to prompt_ids
        if initial_prompt:
            prompt_ids = (
                self.processor.tokenizer(
                    initial_prompt, return_tensors="pt", add_special_tokens=False
                )
                .input_ids[0]
                .to(self.device)
            )
            generate_kwargs["prompt_ids"] = prompt_ids

        result = self.pipe(
            {"array": audio_array, "sampling_rate": _RATE},
            chunk_length_s=30,
            stride_length_s=5,
            generate_kwargs=generate_kwargs,
        )

        return result["text"].strip()  # type: ignore[index]
