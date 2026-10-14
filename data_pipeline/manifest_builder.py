"""
Builds the JSONL manifest for the Hindi→English speech dataset.

Each line is one JSON record:
  {
    "id": "rec-000001",
    "audio": "audio/rec-000001.wav",
    "duration": 4.32,
    "speaker_id": "spk-edge-male",          (optional)
    "source": {
      "corpus": "synthetic-edge-tts",
      "source_file": "audio/rec-000001.wav",
      "start": 0.0,
      "end": 4.32,
      "synthetic": true,
      "model": "edge-tts",                   (for synthetic)
      "voice": "hi-IN-MadhurNeural"
    },
    "sentences": [
      {
        "hi_text": "...",
        "hi_start": 0.21,
        "hi_end": 4.10,
        "en_text": "...",
        "en_alt": ["..."]                    (optional)
      }
    ],
    "hi_words": [{"w": "मैंने", "start": 0.21, "end": 0.55}, ...],
    "translation_method": "Helsinki-NLP/opus-mt-hi-en"
  }
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class WordEntry:
    w: str
    start: float
    end: float


@dataclass
class SentenceEntry:
    hi_text: str
    hi_start: float
    hi_end: float
    en_text: str
    en_alt: Optional[List[str]] = None


@dataclass
class SourceInfo:
    corpus: str
    source_file: str
    start: float
    end: float
    synthetic: bool
    model: Optional[str] = None
    voice: Optional[str] = None


@dataclass
class ManifestRecord:
    id: str
    audio: str
    duration: float
    source: SourceInfo
    sentences: List[SentenceEntry]
    hi_words: List[WordEntry]
    translation_method: str
    speaker_id: Optional[str] = None


class ManifestBuilder:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self._records: List[ManifestRecord] = []

    @staticmethod
    def make_record_id(index: int) -> str:
        return f"rec-{index:06d}"

    def add_record(self, record: ManifestRecord):
        self._records.append(record)

    def save(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            for record in self._records:
                f.write(json.dumps(self._to_dict(record), ensure_ascii=False) + "\n")
        print(f"Saved {len(self._records)} records to {self.manifest_path}")

    def validate(self) -> List[str]:
        """Return a list of format errors (empty = valid)."""
        errors: List[str] = []
        required_top = ["id", "audio", "duration", "source", "sentences", "hi_words", "translation_method"]
        required_source = ["corpus", "source_file", "start", "end", "synthetic"]
        for i, record in enumerate(self._records):
            d = self._to_dict(record)
            for field in required_top:
                if field not in d:
                    errors.append(f"Record {i} ({d.get('id', '?')}): missing '{field}'")
            for field in required_source:
                if field not in d.get("source", {}):
                    errors.append(f"Record {i}: missing source.{field}")
            if not d.get("sentences"):
                errors.append(f"Record {i}: empty sentences list")
            if not d.get("hi_words"):
                errors.append(f"Record {i}: empty hi_words list")
        return errors

    def stats(self) -> dict:
        total_duration = sum(r.duration for r in self._records)
        return {
            "records": len(self._records),
            "total_hours": round(total_duration / 3600, 4),
            "avg_duration_s": round(total_duration / max(len(self._records), 1), 2),
            "synthetic": sum(1 for r in self._records if r.source.synthetic),
            "real": sum(1 for r in self._records if not r.source.synthetic),
        }

    @staticmethod
    def _to_dict(record: ManifestRecord) -> dict:
        source_dict = {
            "corpus": record.source.corpus,
            "source_file": record.source.source_file,
            "start": round(record.source.start, 3),
            "end": round(record.source.end, 3),
            "synthetic": record.source.synthetic,
        }
        if record.source.model:
            source_dict["model"] = record.source.model
        if record.source.voice:
            source_dict["voice"] = record.source.voice

        d = {
            "id": record.id,
            "audio": record.audio,
            "duration": round(record.duration, 3),
            "source": source_dict,
            "sentences": [
                {
                    "hi_text": s.hi_text,
                    "hi_start": round(s.hi_start, 3),
                    "hi_end": round(s.hi_end, 3),
                    "en_text": s.en_text,
                    **({"en_alt": s.en_alt} if s.en_alt else {}),
                }
                for s in record.sentences
            ],
            "hi_words": [
                {"w": w.w, "start": round(w.start, 3), "end": round(w.end, 3)}
                for w in record.hi_words
            ],
            "translation_method": record.translation_method,
        }
        if record.speaker_id:
            d["speaker_id"] = record.speaker_id
        return d
