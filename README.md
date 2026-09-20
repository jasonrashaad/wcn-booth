# wcn-booth

A self-interview rig. A "Coach" — built from my own longitudinal data — asks the
questions; I answer on camera; the whole thing is recorded in OBS with the Coach in one
window and me in the other, Space Ghost Coast to Coast style. Clips go to
[pressroom.whatcomesnextllc.ai](https://pressroom.whatcomesnextllc.ai) and YouTube.

Everything runs on hardware I own. The corpus, the persona, the questions, the voice, the
recording — none of it leaves the house until I choose a clip to publish.

## What to liken it to

The honest comparison is **NotebookLM's Audio Overview** — the feature that turns your
documents into a two-host podcast. It is the exact shape: source material → an AI
personality with a point of view → spoken audio. The difference fits in one sentence:

> Like NotebookLM's podcast feature, except the source material is my own longitudinal
> data, the host is on camera, I'm the guest, and none of it leaves my house.

The last clause is the part that's mine.

And for the record: my reference points for this stuff are Dragon NaturallySpeaking and
downloading TomTom voices. That is fine as a story about *how far the same problem has
come* — a GPS voice in 2006 was a voice actor in a booth for weeks plus a database of
chopped-up syllables; today the voice is a ~500 KB style vector and the text is the trivial
part. It is only a tell if you use it as a reference point for what is *current*. Hence
this section.

## Why lo-fi

Max Headroom was Matt Frewer in prosthetics with video effects — no CGI. Space Ghost was
Hanna-Barbera cels with a handful of mouth positions. The faithful build is three mouth
frames keyed to audio loudness plus a CRT/glitch effects chain, rendered in a browser
page that OBS captures as a Browser Source. There is **no lip-sync ML** in this repo and
there will not be. The only ML in the booth is the voice and the questions.

## How it works

```
PERSONA   (wcn-coach, private)  the record ──► jason-model.md ──► jason-model.public.md
                                              + coach.md          (how the Coach asks)

PREP      topic ──► questions.json ──► qNN.wav (Kokoro TTS) ──► qNN.timeline.json (mouth)

RECORD    booth/director.py (:8788) serves avatar.html to an OBS Browser Source
          hotkey ──► next question plays, mouth flaps, question shows as a lower-third
          OBS: [Coach] [me] [set]  ──►  ~/Movies/booth/

POST      episode ──► MediaCMS (transcode, catalog) ──► Whisper ──► transcript + captions
                  ──► cut list ──► clips ──► playlist

PUBLISH   playlist ──► same-origin <video> on the press room (+ a link to YouTube; no iframe)
```

## Layout

| Path | What |
|---|---|
| *(wcn-coach, private)* | The persona: corpus adapters, the map/reduce builder, the Coach's register, and the `--public` derivation booth reads. Moved out 2026-09-20; `prep/ask.py` finds it via `COACH_REPO` |
| `prep/ask.py` | Topic → questions, one per script-picked observation, with a validator for tics |
| `prep/voice.py` · `timeline.py` | Kokoro (the cast lives here) and the wav → mouth track |
| `booth/director.py` · `avatar.html` | The `:8788` jukebox and the Coach: the `?` mark on a teal CRT, the dot is the mouth |
| `booth/obs/` | The OBS profile + scene collection, exported for rebuild |
| `docs/phase0.md` | The three unknowns, measured |

Not in the repo, by design: the persona (it lives in wcn-coach, a private repo) and `episodes/`.

## Status

**Phases 0–2 done (2026-09-18).** The rig records; the Coach asks. Episode one is
voiced. The Coach asks and does not listen — questions are generated and reviewed before
the session, on purpose. A reactive Coach (Whisper → the brief → a follow-up) is the
next design step, not a limitation of the parts: every stage of it already runs.

Not built: cutting clips, and the press room's video post. Persona refinement is its own
track.

## Running it

```bash
python3 booth/director.py episodes/<episode>     # http://127.0.0.1:8788/
curl -s http://127.0.0.1:8788/api/next           # bind this to a hotkey
```

TTS needs Python ≥ 3.10 (`python3.12 -m venv .venv && .venv/bin/pip install kokoro soundfile`).
Everything else is the standard library.

## Related

- [`wcn-commandcenter`](https://github.com/jasonrashaad/wcn-commandcenter) — the office
  TV channel; the director here is the same shape as its state server on purpose.
- [`wcn-coach`](https://github.com/jasonrashaad/wcn-coach) (private) — the persona: the
  record, the builder, the Coach that answers instead of asks. Booth reads its public
  file and nothing else. The transcription service is a private repo too; only the
  measurements are reproduced here.

## License

Apache 2.0 — see `LICENSE`. Copyright 2026 Jason Rashaad. The code is the shell; the
corpus, the persona, and the episodes are not in this repo and are not licensed to
anyone.
