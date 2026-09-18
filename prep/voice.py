#!/usr/bin/env python3
"""
voice.py — give the Coach a voice: questions.json → qNN.wav + qNN.timeline.json

    .venv/bin/python prep/voice.py episodes/<ep>                 # default voice
    .venv/bin/python prep/voice.py episodes/<ep> --voice bm_george
    .venv/bin/python prep/voice.py --list                        # the voice menu
    .venv/bin/python prep/voice.py --audition "So. Tell me about Sundays."   # every voice, one line

Needs the .venv (Kokoro wants Python ≥ 3.10; the system 3.9 will not do). The timeline
step is stdlib and lives in prep/timeline.py so the director never needs the venv.

Kokoro-82M: Apache-2.0, ~1.9× realtime on an M5's CPU, no GPU needed. A voice is a small
style vector, not a model — swapping one is a one-word change, which is the whole point.
It is not a voice cloner and it is not expressive on its own; it says what is written,
evenly. That evenness is what a broadcast-processed voice sounds like anyway.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 24000
DEFAULT_VOICE = "am_michael"

# The menu. Prefix = accent + gender: a = American, b = British; f/m = female/male.
# Grades are Kokoro's own quality ratings from its voice list (A best). Notes are ours.
VOICES = {
    "am_michael": ("B",  "American male, mid-range, plain. The Phase 0 probe. A newsreader who has seen things."),
    "am_fenrir":  ("C+", "American male, deeper, slower. Closest to a late-night host."),
    "am_puck":    ("C+", "American male, lighter, quicker. Slightly amused by default."),
    "am_adam":    ("F+", "American male, flat and synthetic-sounding. The most 'computer' of the set — on brand, if that is the brand."),
    "am_echo":    ("D",  "American male, soft. Reads as tired."),
    "am_eric":    ("D",  "American male, neutral."),
    "am_liam":    ("D",  "American male, younger."),
    "am_onyx":    ("D",  "American male, low and even."),
    "bm_george":  ("C",  "British male, measured. The interviewer who has done this before."),
    "bm_fable":   ("C",  "British male, warmer, storyteller cadence."),
    "bm_lewis":   ("D+", "British male, brisk."),
    "bm_daniel":  ("D",  "British male, clipped."),
    "af_heart":   ("A",  "American female, Kokoro's best voice. Warm and clear."),
    "af_bella":   ("A-", "American female, bright."),
    "af_nicole":  ("B-", "American female, quiet, close-mic — an ASMR register."),
    "af_sarah":   ("C+", "American female, steady."),
    "af_sky":     ("C-", "American female, light."),
    "bf_emma":    ("B-", "British female, composed."),
    "bf_isabella":("C",  "British female, warm."),
}


def synth(pipe, text, voice):
    import numpy as np
    chunks = [audio for _, _, audio in pipe(text, voice=voice)]
    return np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")


def write_wav(path, audio):
    import soundfile as sf
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="?", help="episode directory holding questions.json")
    ap.add_argument("--voice", default=DEFAULT_VOICE, choices=sorted(VOICES))
    ap.add_argument("--list", action="store_true", help="print the voice menu and exit")
    ap.add_argument("--audition", metavar="TEXT", help="render TEXT in every voice to episodes/_audition/")
    ap.add_argument("--force", action="store_true", help="re-render wavs that already exist")
    args = ap.parse_args()

    if args.list:
        for v, (grade, note) in VOICES.items():
            print(f"{v:13s} {grade:3s} {note}")
        return

    from kokoro import KPipeline   # slow import; only after --list
    pipe = KPipeline(lang_code="a" if not args.voice.startswith("b") else "b")

    if args.audition:
        out = HERE.parent / "episodes" / "_audition"
        out.mkdir(parents=True, exist_ok=True)
        for v in VOICES:
            p = KPipeline(lang_code="b" if v.startswith("b") else "a") if v[0] != args.voice[0] else pipe
            write_wav(out / f"{v}.wav", synth(p, args.audition, v))
            print(f"  {v}.wav")
        print(f"→ {out}   (afplay each, or: for f in {out}/*.wav; do echo $f; afplay $f; done)")
        return

    if not args.episode:
        sys.exit("episode directory required (or --list / --audition)")
    ep = Path(args.episode)
    doc = json.loads((ep / "questions.json").read_text(encoding="utf-8"))
    todo = []
    for q in doc["questions"]:
        wav = ep / q["audio"]
        if wav.is_file() and not args.force:
            continue
        write_wav(wav, synth(pipe, q["text"], args.voice))
        todo.append(wav)
        print(f"  {q['id']}  {wav.stat().st_size//1024:4d} KB  {q['text'][:70]}")
    doc["voice"] = args.voice
    (ep / "questions.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    if todo:
        subprocess.run([sys.executable, str(HERE / "timeline.py"), *map(str, todo)], check=True)
    print(f"{len(todo)} rendered in voice {args.voice}; {len(doc['questions'])-len(todo)} already present")


if __name__ == "__main__":
    main()
