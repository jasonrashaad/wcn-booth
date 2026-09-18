#!/usr/bin/env python3
"""
build-persona.py — a habit model of one person, derived from their own transcripts.

    python3 persona/build-persona.py                 # map + reduce → persona/jason-model*.md
    python3 persona/build-persona.py --map-only      # just the per-batch extraction
    python3 persona/build-persona.py --reduce-only   # rebuild the documents from saved extractions
    python3 persona/build-persona.py --limit 3       # first N batches, for a smoke test

Reads persona/corpus/*.json (the transcript JSON the Whisper service writes: `text`,
`segments`, `audio_s`), orders clips by the date in their filename, and runs two passes
against a local Ollama:

  MAP     batches of ~BATCH_WORDS words → structured observations, each tagged with the
          clip it came from, the kind of thing it is, a verbatim quote, and whether it is
          safe to raise on camera. Saved to persona/extract/<batch>.json so a reduce can
          be re-run without paying for the map again.
  REDUCE  all observations → two Markdown documents:
            persona/jason-model.md          everything. PRIVATE. never leaves this disk.
            persona/jason-model.public.md   only observations tagged public. This is the
                                            ONLY file prep/ask.py may read, and it is
                                            meant to be edited by hand before it is used.

Stdlib only. Nothing here leaves the LAN: the corpus is on this disk, inference is on
the workbench, and the outputs are gitignored.

WHY TWO DOCUMENTS
    The corpus is a diary, not a habit log. A Coach that asks questions for YouTube must
    not have the diary — it may have what the person has read and chosen to keep. The
    model's public/private tag is a first pass; the person's edit is the real fence.

WHY OBSERVATIONS CITE A CLIP
    "No judgment, just data" only holds if every claim can be traced to something that
    was actually said. A question that cites a date is a mirror; one that does not is
    an opinion.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
EXTRACT = HERE / "extract"
OLLAMA = os.environ.get("BOOTH_OLLAMA", "http://192.168.1.118:11434")
MODEL = os.environ.get("BOOTH_MODEL", "qwen3:30b")
BATCH_WORDS = 1800          # ~2.5k tokens of transcript per map call
NUM_CTX = 16384

KINDS = ["intent", "action", "theme", "cadence", "phrase", "belief", "number", "person", "tension"]

MAP_SYSTEM = """You are reading transcripts of one person's private voice memos. They are
unedited, profane, and sometimes include other people talking. Your job is to extract
OBSERVATIONS about the primary speaker — things a coach who has known them for a year
would notice — and nothing else.

Each observation has:
- kind: one of
    intent   — something they say they WILL do, or want to do
    action   — something they report they DID (or did not do)
    theme    — a subject they keep returning to
    cadence  — a named ritual, routine, day-of-week pattern, or repeated practice
    phrase   — a turn of phrase that is distinctly theirs (verbatim)
    belief   — a stated principle or rule they hold themselves to
    number   — a concrete figure, date, count, or measurement they state
    person   — a named person and the role they play (ALWAYS private)
    tension  — a contradiction between two things they said, or between intent and action
- text: the observation in plain third person, one sentence, specific.
- quote: a short verbatim excerpt from the transcript that supports it (their words).
- clip: the clip id given in the transcript header.
- public: true ONLY if a stranger could hear this raised on camera without it costing
  the speaker anything — habits, work, training, food, routines, projects, stated
  principles, jokes about themselves. false for sex, relationships, named people, family
  conflict, substances, money trouble, health details, legal matters, or anything said
  about a third party. When unsure, false.

Rules: extract only what is actually said. Do not diagnose, do not soften their language,
do not invent. Skip small talk. Six to twenty observations per batch is normal."""

MAP_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": KINDS},
                    "text": {"type": "string"},
                    "quote": {"type": "string"},
                    "clip": {"type": "string"},
                    "public": {"type": "boolean"},
                },
                "required": ["kind", "text", "quote", "clip", "public"],
            },
        }
    },
    "required": ["observations"],
}

REDUCE_SYSTEM = """You are writing a habit model of one person from a list of dated
observations extracted from their own voice memos. Write it the way a coach who has
known them for a year would write their file: specific, in plain language, no judgment,
no diagnosis, no advice. Their own words are quoted back unsoftened.

Output Markdown with exactly these sections, in this order, each a list of bullets.
Every bullet ends with a citation like (2025-10-07) or (2025-10-07, 2026-07-12) — the
dates of the clips it rests on. Merge duplicates; keep the count honest (say "at least
four times" if it was said four times). Leave a section as "- (nothing in the corpus)"
if there is nothing.

## Who this is, in their own words
## What they keep coming back to
## What they said they would do
## What they said they did
## Named cadences and rituals
## Phrases that are theirs
## What they hold themselves to
## Numbers and dates they stated
## Tensions
"""


def clip_date(name):
    # MediaCMS prefixes a 32-hex hash that can itself contain "2019…" — strip it first.
    name = re.sub(r"^[0-9a-f]{32}\.", "", name)
    m = re.search(r"(20\d{2})[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])", name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "undated"


def load_corpus():
    clips = []
    for f in sorted(glob.glob(str(CORPUS / "*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        text = (d.get("text") or "").strip()
        if len(text.split()) < 8:
            continue
        stem = Path(f).stem
        clips.append({"id": stem, "date": clip_date(stem), "words": len(text.split()),
                      "audio_s": d.get("audio_s"), "text": text})
    clips.sort(key=lambda c: (c["date"] == "undated", c["date"], c["id"]))
    return clips


def batches(clips):
    cur, n = [], 0
    for c in clips:
        if cur and n + c["words"] > BATCH_WORDS:
            yield cur
            cur, n = [], 0
        cur.append(c); n += c["words"]
    if cur:
        yield cur


def ollama(system, user, schema=None, temperature=0.2, timeout=600):
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_ctx": NUM_CTX},
    }
    if schema:
        body["format"] = schema
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out["message"]["content"], out


def run_map(clips, limit=None):
    EXTRACT.mkdir(exist_ok=True)
    all_batches = list(batches(clips))
    if limit:
        all_batches = all_batches[:limit]
    print(f"map: {len(clips)} clips → {len(all_batches)} batches of ~{BATCH_WORDS} words, model {MODEL}")
    for i, b in enumerate(all_batches, 1):
        out = EXTRACT / f"{i:03d}.json"
        if out.is_file():
            continue
        user = "\n\n".join(f"### clip {c['id']} (date {c['date']}, {c['audio_s']:.0f}s)\n{c['text']}" for c in b)
        t0 = time.time()
        content, raw = ollama(MAP_SYSTEM, user, schema=MAP_SCHEMA)
        try:
            obs = json.loads(content)["observations"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  batch {i}: bad JSON ({e}); saved raw to {out.with_suffix('.raw')}")
            out.with_suffix(".raw").write_text(content)
            continue
        ids = {c["id"] for c in b}
        dates = {c["id"]: c["date"] for c in b}
        kept = []
        for o in obs:
            if o["clip"] not in ids:      # the model may paraphrase the id; try a prefix match
                m = [cid for cid in ids if cid.startswith(o["clip"][:12])]
                if not m:
                    continue
                o["clip"] = m[0]
            o["date"] = dates[o["clip"]]
            kept.append(o)
        out.write_text(json.dumps({"batch": i, "clips": sorted(ids), "observations": kept}, indent=1))
        print(f"  batch {i:3d}/{len(all_batches)}: {len(b):2d} clips, {len(kept):2d} obs "
              f"({sum(o['public'] for o in kept)} public) in {time.time()-t0:.0f}s "
              f"[{raw.get('eval_count',0)} tok]")


def load_observations():
    obs = []
    for f in sorted(EXTRACT.glob("*.json")):
        obs.extend(json.loads(f.read_text())["observations"])
    return obs


def render_bullets(observations):
    """Feed the reduce model a compact, kind-grouped, date-ordered listing."""
    by_kind = defaultdict(list)
    for o in observations:
        by_kind[o["kind"]].append(o)
    parts = []
    for k in KINDS:
        if not by_kind[k]:
            continue
        parts.append(f"# {k} ({len(by_kind[k])})")
        for o in sorted(by_kind[k], key=lambda o: o["date"]):
            parts.append(f"- ({o['date']}) {o['text']}  — \"{o['quote']}\"")
    return "\n".join(parts)


def run_reduce(observations, label):
    listing = render_bullets(observations)
    print(f"reduce [{label}]: {len(observations)} observations, {len(listing.split())} words in")
    t0 = time.time()
    content, raw = ollama(REDUCE_SYSTEM, listing, temperature=0.3, timeout=1800)
    print(f"  {raw.get('eval_count',0)} tok out in {time.time()-t0:.0f}s")
    return content.strip()


def header(label, observations, clips):
    dates = sorted({o["date"] for o in observations if o["date"] != "undated"})
    kinds = Counter(o["kind"] for o in observations)
    return (f"<!-- jason-model [{label}] — generated {time.strftime('%Y-%m-%d %H:%M')} by "
            f"persona/build-persona.py, model {MODEL}. {len(clips)} clips, {len(observations)} "
            f"observations ({', '.join(f'{k} {n}' for k, n in kinds.most_common())}). "
            f"Corpus spans {dates[0] if dates else '?'} → {dates[-1] if dates else '?'}. "
            f"{'PRIVATE. Never leaves this disk.' if label == 'private' else 'Edit by hand before prep/ask.py uses it.'} -->\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--reduce-only", action="store_true")
    ap.add_argument("--limit", type=int, help="first N batches only")
    args = ap.parse_args()

    clips = load_corpus()
    if not args.reduce_only:
        run_map(clips, args.limit)
    if args.map_only:
        return
    obs = load_observations()
    if not obs:
        sys.exit("no observations in persona/extract/ — run the map first")

    private = run_reduce(obs, "private")
    (HERE / "jason-model.md").write_text(header("private", obs, clips) + private + "\n", encoding="utf-8")
    pub_obs = [o for o in obs if o["public"] and o["kind"] != "person"]
    public = run_reduce(pub_obs, "public")
    (HERE / "jason-model.public.md").write_text(header("public", pub_obs, clips) + public + "\n", encoding="utf-8")
    print(f"wrote persona/jason-model.md ({len(private.split())} words) and "
          f"persona/jason-model.public.md ({len(public.split())} words, {len(pub_obs)}/{len(obs)} observations)")


if __name__ == "__main__":
    main()
