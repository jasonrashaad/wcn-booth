# Phase 0 — the three unknowns, measured 2026-09-17/18

Everything else in the plan is known-good tech. These three were not.

## 1. Kokoro TTS on wcn-macbook (Apple M5, 32 GB) — works, CPU only

- System Python is 3.9.6; Kokoro needs ≥3.10. `.venv` is `/opt/homebrew/bin/python3.12`
  (`brew install python@3.12`). `brew install ffmpeg` too — the Mac had none.
- `pip install kokoro soundfile` pulls torch 2.14 + spaCy `en_core_web_sm` (first run only).
- First run: 57.8 s "load" — that is the model download. Warm: **10.8 s load, 4.6 s to
  synthesize 8.8 s of speech ≈ 1.9× realtime** on CPU. Twelve questions is under a minute.
- Voice in the probe: `am_michael`. Any Kokoro voice is a one-word swap.
- `prep/timeline.py` (stdlib `wave`) → 25 fps closed/mid/open. Kokoro's output is evenly
  compressed, so thresholds are **0.30 / 0.12 of peak RMS** — 0.45 barely ever fired.

## 2. OBS Browser Source audio — page fixed, capture still to be confirmed

- First attempt showed a directory listing: `SimpleHTTPRequestHandler` only auto-serves
  `index.html`, and the page is `avatar.html`. `director.py` now maps `/` → `avatar.html`.
- The OBS source (`wcn-booth`, in the `WCN` scene collection) was configured correctly on
  the first try: `http://127.0.0.1:8788/`, 1280×720, `reroute_audio: true`.
- `/api/next` bumps a `serial`, not just the index, so a repeat of the same question replays.
- **Still to confirm:** the recorded file carries the browser audio. If not, fall back to
  pre-mixing in post off the timeline.

## 3. Big-model headroom on wcn-workbench (RX 7800 XT 16 GB + 30 GB RAM)

- `qwen3:30b` (Qwen3-30B-A3B MoE, Q4, 19.2 GB) pulled in ~4 min at ~90 MB/s.
- **600 tokens in 13.6 s = 44.1 tok/s**, load 7.7 s. 16.0 GB in VRAM, ~3 GB spilled to RAM.
- Same prompt on `qwen2.5:14b`: 44.7 tok/s. **The 30B MoE costs nothing at generation** —
  3B active parameters per token is the reason. Patience is not required; disk is.
- `"think": false` on `/api/generate` did **not** suppress the reasoning preamble in this
  Ollama (v0.24.0) — the response opened "Okay, the user wants…". Use `/api/chat` with
  `think:false`, or prepend `/no_think`, and verify before trusting the output shape.

### Found on the way: `wcn-whisper` had been down for six weeks

`docker compose ps -a` in `/opt/whisper-poc`: `Exited (0) 6 weeks ago`. Clean stop, never
restarted — the container's restart policy does not survive a deliberate `stop`. Every
other container on the box was up 4 weeks. WCN-INFRA said it was resident; it was not.

Bringing it back reproduced the documented contention exactly: with `qwen2.5:14b` still
resident from the benchmark (10 GB), Whisper crash-looped on
`torch.OutOfMemoryError: HIP out of memory … 0 bytes is free`. Unloading Ollama's models
(`keep_alive: 0`) let it start; healthy in under a minute, ~6.4 GB VRAM.

**Order of operations for every episode, then:** transcribe first (Whisper resident),
*then* unload nothing and run the LLM pass only if it fits, else `keep_alive:0` / stop
Whisper first. Ollama drops a model after 5 min idle by default, which is usually enough.

Whisper binds `192.168.1.118:9077`, not loopback — `curl localhost:9077` on the box is
empty by design, not a failure.
