#!/usr/bin/env python3
"""
timeline.py — turn a spoken wav into a mouth timeline the avatar page can play.

    python3 prep/timeline.py episodes/<ep>/q01.wav          # writes q01.timeline.json

Space Ghost, not SadTalker: three mouth states (0 closed, 1 mid, 2 open) sampled
every FRAME_MS, chosen by RMS loudness against two thresholds set relative to the
clip's own peak. Stdlib only — `wave` + `array` — so this runs anywhere, including
inside the director with no venv. Kokoro writes 24 kHz mono 16-bit PCM via soundfile;
anything else mono/16-bit works too.
"""
import array
import json
import math
import sys
import wave
from pathlib import Path

FRAME_MS = 40          # 25 fps — matches the limited-animation feel on purpose
OPEN_AT = 0.30         # fraction of peak RMS → state 2 (Kokoro output is compressed; 0.45 barely fired)
MID_AT = 0.12          # fraction of peak RMS → state 1


def rms_frames(path):
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "mono 16-bit PCM expected"
        rate = w.getframerate()
        pcm = array.array("h", w.readframes(w.getnframes()))
    n = max(1, rate * FRAME_MS // 1000)
    out = []
    for i in range(0, len(pcm), n):
        chunk = pcm[i:i + n]
        out.append(math.sqrt(sum(s * s for s in chunk) / len(chunk)) if chunk else 0.0)
    return rate, out


def timeline(path):
    rate, frames = rms_frames(path)
    peak = max(frames) or 1.0
    states = [2 if f >= peak * OPEN_AT else 1 if f >= peak * MID_AT else 0 for f in frames]
    return {
        "frame_ms": FRAME_MS,
        "duration_ms": len(frames) * FRAME_MS,
        "states": states,
    }


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        src = Path(arg)
        tl = timeline(src)
        dst = src.with_suffix(".timeline.json")
        dst.write_text(json.dumps(tl, separators=(",", ":")), encoding="utf-8")
        hist = [tl["states"].count(s) for s in (0, 1, 2)]
        print(f"{dst.name}: {tl['duration_ms']} ms, frames closed/mid/open = {hist}")


if __name__ == "__main__":
    main()
