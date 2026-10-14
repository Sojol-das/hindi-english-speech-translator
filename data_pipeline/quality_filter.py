"""
Quality checks for dataset records.

Validates:
- Clip duration: 2–60 seconds
- Word timings: ordered, non-negative, inside the clip
- Non-empty transcript and translation
- Rough speech-rate sanity check for TTS artifacts
"""

from dataclasses import dataclass
from typing import List

from data_pipeline.transcriber import WordAlignment

MIN_DURATION = 2.0    # seconds (we allow shorter for synthetic clips)
MAX_DURATION = 60.0   # seconds (hard limit)
MIN_WORDS = 2
MAX_CHAR_RATE = 35.0  # chars/sec — higher than this suggests TTS rushing/skipping


@dataclass
class QualityReport:
    passed: bool
    issues: List[str]

    def __str__(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL: " + "; ".join(self.issues)


class QualityFilter:
    def check(
        self,
        duration: float,
        transcript: str,
        words: List[WordAlignment],
        translation: str,
    ) -> QualityReport:
        issues: List[str] = []

        # Duration bounds
        if duration < MIN_DURATION:
            issues.append(f"Too short: {duration:.2f}s < {MIN_DURATION}s")
        if duration > MAX_DURATION:
            issues.append(f"Too long: {duration:.2f}s > {MAX_DURATION}s")

        # Transcript
        if not transcript.strip():
            issues.append("Empty transcript")

        # Word count
        if len(words) < MIN_WORDS:
            issues.append(f"Too few words: {len(words)}")

        # Word timing validity
        for i, w in enumerate(words):
            if w.start < 0:
                issues.append(f"Word[{i}] '{w.word}': negative start {w.start}")
            if w.end <= w.start:
                issues.append(f"Word[{i}] '{w.word}': end {w.end} <= start {w.start}")
            if w.end > duration + 0.15:  # 150 ms tolerance for rounding
                issues.append(
                    f"Word[{i}] '{w.word}': end {w.end} > clip duration {duration:.2f}"
                )
            if i > 0 and w.start < words[i - 1].end - 0.05:
                issues.append(
                    f"Word[{i}] '{w.word}': overlaps previous word"
                )

        # Translation
        if not translation.strip():
            issues.append("Empty translation")

        # Speech-rate sanity (catches TTS word-skipping artifacts)
        if transcript and duration > 0:
            rate = len(transcript) / duration
            if rate > MAX_CHAR_RATE:
                issues.append(f"Abnormal char rate: {rate:.1f} chars/s (max {MAX_CHAR_RATE})")

        return QualityReport(passed=len(issues) == 0, issues=issues)
