"""
Hindi → English translation using Helsinki-NLP/opus-mt-hi-en (MarianMT).

This is a compact (~300 MB) model that runs on CPU. For production quality
on Hinglish / code-switched text, IndicTrans2 or an LLM is better — swap
in by subclassing BaseTranslator and implementing translate_batch().
"""

from abc import ABC, abstractmethod
from typing import List


class BaseTranslator(ABC):
    @abstractmethod
    def translate(self, hindi_text: str) -> str: ...

    @abstractmethod
    def translate_batch(self, texts: List[str]) -> List[str]: ...

    def translate_with_alternatives(
        self, hindi_text: str, num_alt: int = 1
    ) -> List[str]:
        """Return [best_translation, ...alternatives]. Override for beam diversity."""
        return [self.translate(hindi_text)]


class MarianTranslator(BaseTranslator):
    MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"

    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer
        import torch

        print(f"Loading translation model {self.MODEL_NAME}...")
        self.tokenizer = MarianTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = MarianMTModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        self._torch = torch

    def translate(self, hindi_text: str) -> str:
        return self.translate_batch([hindi_text])[0]

    def translate_batch(self, texts: List[str]) -> List[str]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with self._torch.no_grad():
            outputs = self.model.generate(**inputs, num_beams=4, max_length=512)
        return [
            self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs
        ]

    def translate_with_alternatives(
        self, hindi_text: str, num_alt: int = 1
    ) -> List[str]:
        num_seq = num_alt + 1
        inputs = self.tokenizer(
            [hindi_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=max(4, num_seq),
                num_return_sequences=num_seq,
                max_length=512,
            )
        return [
            self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs
        ]


# Default export
HindiEnglishTranslator = MarianTranslator
