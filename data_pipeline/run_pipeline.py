#!/usr/bin/env python3
"""
Data pipeline: synthetic Hindi speech → transcript + alignment + translation → JSONL manifest.

Usage:
  python data_pipeline/run_pipeline.py --output-dir dataset_output --num-samples 20
  python data_pipeline/run_pipeline.py --input-file sample_data/hindi_sentences.txt --output-dir dataset_output
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_sentences(filepath: str) -> list[str]:
    sentences = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sentences.append(line)
    return sentences


DEFAULT_SENTENCES = [
    "मैंने कल अपना पुराना बैंक खाता बंद करवा दिया।",
    "मुझे तीन लाख का loan चाहिए, EMI कितनी होगी?",
    "क्या आप मुझे नजदीकी hospital का address बता सकते हैं?",
    "मेरे account में पचास हजार रुपये transfer कर दो।",
    "अगर कल rain नहीं हुई तो हम picnic पर जाएंगे।",
    "नहीं, मैंने कभी नहीं कहा कि मैं वहां जाऊंगा।",
    "यह mobile बहुत अच्छा है, इसकी battery बहुत लंबे समय तक चलती है।",
    "मेरा phone number नौ आठ सात छह पांच चार तीन दो एक शून्य है।",
    "इस साल हमारी company का profit पांच करोड़ रुपये था।",
    "राहुल गुप्ता कल दिल्ली से मुंबई जा रहे हैं।",
    "appointment पंद्रह नवंबर को तीन बजे है।",
    "वो... वो Sanjay है, नहीं नहीं, Suresh है।",
]


def run_pipeline(output_dir: str, sentences: list[str], voice: str = "male", whisper_model: str = "small"):
    from data_pipeline.audio_processor import AudioProcessor
    from data_pipeline.transcriber import HindiTranscriber
    from data_pipeline.translator import HindiEnglishTranslator
    from data_pipeline.tts_generator import TTSGenerator
    from data_pipeline.quality_filter import QualityFilter
    from data_pipeline.manifest_builder import (
        ManifestBuilder, ManifestRecord, SentenceEntry, WordEntry, SourceInfo,
    )

    out = Path(output_dir)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Initializing models ===")
    tts = TTSGenerator()
    transcriber = HindiTranscriber(model_size=whisper_model)
    translator = HindiEnglishTranslator()
    qf = QualityFilter()
    builder = ManifestBuilder(output_dir)

    print(f"\n=== Processing {len(sentences)} sentences ===\n")
    passed = 0
    failed = 0

    for i, hindi_text in enumerate(tqdm(sentences, desc="Pipeline")):
        record_id = ManifestBuilder.make_record_id(i)
        audio_path = str(audio_dir / f"{record_id}.wav")
        rel_audio = f"audio/{record_id}.wav"

        # Step 1: Synthetic Hindi TTS
        try:
            duration = tts.generate_hindi(hindi_text, audio_path, voice=voice)
        except Exception as e:
            tqdm.write(f"[SKIP] {record_id}: TTS failed — {e}")
            failed += 1
            continue

        # Step 2: Transcribe with word timestamps
        try:
            tx = transcriber.transcribe(audio_path)
        except Exception as e:
            tqdm.write(f"[SKIP] {record_id}: Transcription failed — {e}")
            failed += 1
            continue

        if not tx.text.strip():
            tqdm.write(f"[SKIP] {record_id}: Empty transcript")
            failed += 1
            continue

        # Step 3: Translate (best + 1 alternative)
        try:
            translations = translator.translate_with_alternatives(tx.text, num_alt=1)
        except Exception as e:
            tqdm.write(f"[SKIP] {record_id}: Translation failed — {e}")
            failed += 1
            continue

        en_text = translations[0]
        en_alt = translations[1:] if len(translations) > 1 else []

        # Step 4: Quality check
        report = qf.check(
            duration=duration,
            transcript=tx.text,
            words=tx.words,
            translation=en_text,
        )
        if not report.passed:
            tqdm.write(f"[FAIL] {record_id}: {report}")
            failed += 1
            continue

        # Step 5: Build manifest record
        hi_start = tx.words[0].start if tx.words else 0.0
        hi_end = tx.words[-1].end if tx.words else duration
        voice_name = "hi-IN-MadhurNeural" if voice == "male" else "hi-IN-SwaraNeural"

        record = ManifestRecord(
            id=record_id,
            audio=rel_audio,
            duration=round(duration, 3),
            speaker_id=f"spk-edge-{voice}",
            source=SourceInfo(
                corpus="synthetic-edge-tts",
                source_file=rel_audio,
                start=0.0,
                end=round(duration, 3),
                synthetic=True,
                model="edge-tts",
                voice=voice_name,
            ),
            sentences=[
                SentenceEntry(
                    hi_text=tx.text,
                    hi_start=hi_start,
                    hi_end=hi_end,
                    en_text=en_text,
                    en_alt=en_alt if en_alt else None,
                )
            ],
            hi_words=[WordEntry(w=w.word, start=w.start, end=w.end) for w in tx.words],
            translation_method="Helsinki-NLP/opus-mt-hi-en (MarianMT, beam=4)",
        )
        builder.add_record(record)
        passed += 1

        tqdm.write(f"[OK]   {record_id}  {duration:.1f}s  \"{tx.text[:50]}...\"")
        tqdm.write(f"         EN: \"{en_text[:60]}\"")

    # Save and validate
    print("\n=== Saving manifest ===")
    builder.save()

    errors = builder.validate()
    if errors:
        print(f"\nFormat validation ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    else:
        print("Format validation: PASSED")

    stats = builder.stats()
    print(f"\n=== Dataset Statistics ===")
    print(f"  Records:      {stats['records']}")
    print(f"  Total hours:  {stats['total_hours']:.4f}")
    print(f"  Avg duration: {stats['avg_duration_s']:.2f}s")
    print(f"  Synthetic:    {stats['synthetic']}")
    print(f"  Real:         {stats['real']}")
    print(f"  Passed QC:    {passed}")
    print(f"  Failed/Skipped: {failed}")
    print(f"\nOutput: {out.resolve()}/")


def main():
    parser = argparse.ArgumentParser(
        description="Hindi TTS → ASR → translation → JSONL dataset pipeline"
    )
    parser.add_argument(
        "--input-file",
        help="Text file with one Hindi sentence per line (# = comment)",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset_output",
        help="Directory to write audio + manifest.jsonl",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
        help="How many sentences to process (cycles through DEFAULT_SENTENCES if no --input-file)",
    )
    parser.add_argument(
        "--voice",
        choices=["male", "female"],
        default="male",
        help="Edge TTS Hindi voice",
    )
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
        help="Whisper model size (small runs on CPU in ~10–30s per clip)",
    )
    args = parser.parse_args()

    if args.input_file:
        sentences = load_sentences(args.input_file)
        print(f"Loaded {len(sentences)} sentences from {args.input_file}")
    else:
        sentences = (DEFAULT_SENTENCES * 10)[: args.num_samples]

    run_pipeline(args.output_dir, sentences, voice=args.voice, whisper_model=args.whisper_model)


if __name__ == "__main__":
    main()
