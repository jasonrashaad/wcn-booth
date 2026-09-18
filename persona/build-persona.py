#!/usr/bin/env python3
"""
build-persona.py — a habit model of one person, derived from their own transcripts.

    python3 persona/build-persona.py                 # map + reduce → persona/jason-model*.md
    python3 persona/build-persona.py --map-only      # just the per-batch extraction
    python3 persona/build-persona.py --reduce-only   # rebuild jason-model.md from saved extractions
    python3 persona/build-persona.py --reduce-only --only "Tensions"   # redo ONE section in place
    python3 persona/build-persona.py --public        # derive jason-model.public.md from the EDITED private file
    python3 persona/build-persona.py --limit 3       # first N batches, for a smoke test

Reads persona/corpus/*.json (the transcript JSON the Whisper service writes: `text`,
`segments`, `audio_s`), orders clips by the date in their filename, and runs two passes
against a local Ollama:

  MAP     batches of ~BATCH_WORDS words → structured observations, each tagged with the
          clip it came from, the kind of thing it is, a verbatim quote, and whether it is
          safe to raise on camera. Saved to persona/extract/<batch>.json so a reduce can
          be re-run without paying for the map again.
  REDUCE  observations → persona/jason-model.md, one section at a time (546 observations
          do not fit one context window). Every bullet is tagged [public] or [private].
          PRIVATE. Never leaves this disk. This is the file the person reads and edits:
          flip tags, delete lines, fix what the model misread.
  PUBLIC  --public derives persona/jason-model.public.md from the edited private file by
          keeping only [public] bullets. A plain filter, no model — so nothing private can
          be paraphrased across the fence. This is the ONLY file prep/ask.py may read.

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
CLOSE_S = 12.0              # the saved line: the final seconds before the save key (see closers.py)
NUM_CTX = 24576

KINDS = ["moment", "intent", "action", "theme", "cadence", "phrase", "belief", "number", "person", "tension"]

MAP_SYSTEM = """You are reading one person's private voice journal. Each entry is a clip they
saved by pressing a key when they heard themselves say something worth keeping — so
the END of each clip (marked SAVED LINE) is the reason the clip exists, and the rest is
the lead-up. Entries are unedited, profane, and sometimes include other people talking.

Your job is to extract OBSERVATIONS about the primary speaker — what a coach who has
known them for a year would write in their file. WHEN matters more than what: the
speaker will recall where they were in their life from the date, and the meaning of
what they said depends on it. Every observation is anchored to its clip and date.

Each observation has:
- kind: one of
    moment   — what was GOING ON at that date: a job, a move, a deadline, a decision, a
               fight, a start or an end. The timeline. Extract these generously.
    intent   — something they say they will do or want to do
    action   — something they report they did (or did not do)
    theme    — a subject they keep returning to
    cadence  — a named ritual, routine, day-of-week pattern, or repeated practice
    phrase   — a turn of phrase that is distinctly theirs (verbatim)
    belief   — a stated principle or rule they hold themselves to
    number   — a concrete figure, date, count, or measurement they state
    person   — a named person and the role they play (ALWAYS private)
    tension  — a contradiction between two things they said, or between intent and action
- text: the observation in plain third person, one sentence, specific. If it comes from
  the SAVED LINE, begin it with "Saved:".
- quote: a short verbatim excerpt that supports it (their words, unsoftened).
- clip: the clip id given in the entry header.
- public: this file is for the speaker's OWN on-camera self-interview, and they are not
  shy about their own life. So public is true by default. It is false ONLY when the
  observation names, describes, or exposes SOMEONE ELSE — a partner, family, a friend,
  a co-worker, a client — or tells a third party's story. Protect other people, not
  the speaker.

Rules: extract only what is actually said. Do not diagnose, do not soften their
language, do not invent. Skip small talk. Eight to twenty-five observations per batch
is normal; the SAVED LINE of every clip should yield at least one."""

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

REDUCE_SYSTEM = """You are writing ONE SECTION of a habit model of one person, from dated
observations extracted from their own voice journal. Write it the way a coach who has
known them for a year would write their file: specific, plain language, no judgment, no
diagnosis, no advice. Their own words are quoted back unsoftened. WHEN matters more
than what: lead a bullet with the time where you can ("Early October 2025: …") and never
merge observations from different months into one bullet — the same thought in
October and again in July is two bullets, because the person will read them as two
different lives.

Return a list of bullets. Each bullet is one specific point that merges duplicate
observations; keep the count honest ("at least four times"). `dates` lists the dates of
every clip it rests on. `public` is true only if EVERY observation it rests on was marked
public. Fewer bullets than observations; never more. Order by earliest date."""

REDUCE_SCHEMA = {
    "type": "object",
    "properties": {
        "bullets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "dates": {"type": "array", "items": {"type": "string"}},
                    "public": {"type": "boolean"},
                },
                "required": ["text", "dates", "public"],
            },
        }
    },
    "required": ["bullets"],
}

# Section → the observation kinds it is built from. A kind may feed more than one section.
SECTIONS = [
    ("What was going on, when",           ["moment"]),
    ("Who this is, in their own words",   ["theme", "belief", "phrase"]),
    ("What they keep coming back to",     ["theme"]),
    ("What they said they would do",      ["intent"]),
    ("What they said they did",           ["action"]),
    ("Named cadences and rituals",        ["cadence"]),
    ("Phrases that are theirs",           ["phrase"]),
    ("What they hold themselves to",      ["belief"]),
    ("Numbers and dates they stated",     ["number"]),
    ("People",                            ["person"]),
    ("Tensions",                          ["tension"]),
]
TAG = re.compile(r"\[(public|private)\]\s*$")


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
        segs = d.get("segments") or []
        end = segs[-1].get("end", 0.0) if segs else 0.0
        closer = " ".join(sg.get("text", "").strip() for sg in segs if sg.get("end", 0.0) > end - CLOSE_S)
        clips.append({"id": stem, "date": clip_date(stem), "words": len(text.split()),
                      "audio_s": d.get("audio_s") or end, "text": text,
                      "closer": re.sub(r"\s+", " ", closer).strip() or text[-300:]})
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
        user = "\n\n".join(
            f"### clip {c['id']} — DATE {c['date']} — {c['audio_s']:.0f}s\n{c['text']}\n"
            f"SAVED LINE (the last {CLOSE_S:.0f}s, the reason this clip exists): {c['closer']}"
            for c in b)
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
    """Compact, date-ordered listing for one section's kinds."""
    lines = []
    for o in sorted(observations, key=lambda o: (o["date"], o["kind"])):
        vis = "public" if o["public"] else "private"
        lines.append(f"- ({o['date']}) [{o['kind']}, {vis}] {o['text']}  — \"{o['quote']}\"")
    return "\n".join(lines)


def degenerate(bullets, n_obs):
    """Constrained decoding guarantees the shape, not the content. Catch placeholders."""
    if not bullets:
        return True
    texts = [bl["text"].strip().lower() for bl in bullets]
    if any(t in ("text here", "text", "...", "") or t.startswith("bullet") for t in texts):
        return True
    return n_obs >= 10 and len(bullets) < 3


def reduce_section(title, kinds, observations, tries=3):
    sub = [o for o in observations if o["kind"] in kinds]
    if not sub:
        return f"## {title}\n\n- (nothing in the corpus)\n"
    user = f"Section: {title}\n\n{render_bullets(sub)}"
    bullets, raw, t0 = [], {}, time.time()
    for attempt in range(1, tries + 1):
        content, raw = ollama(REDUCE_SYSTEM, user, schema=REDUCE_SCHEMA,
                              temperature=0.3 + 0.2 * (attempt - 1), timeout=1800)
        try:
            bullets = json.loads(content)["bullets"]
        except (json.JSONDecodeError, KeyError):
            bullets = []
        if not degenerate(bullets, len(sub)):
            break
        print(f"  {title}: degenerate output on try {attempt} ({len(bullets)} bullets); retrying warmer")
        bullets = []
    lines = []
    for bl in bullets[: len(sub)]:          # a reduce never expands
        dates = ", ".join(sorted(set(d for d in bl["dates"] if d)))
        tag = "[public]" if bl["public"] else "[private]"
        lines.append(f"- {bl['text'].strip()} ({dates}) {tag}")
    n_pub = sum(1 for l in lines if l.endswith("[public]"))
    print(f"  {title}: {len(sub)} obs → {len(lines)} bullets ({n_pub} public) "
          f"in {time.time()-t0:.0f}s [{raw.get('eval_count',0)} tok]")
    return f"## {title}\n\n" + ("\n".join(lines) if lines else "- (reduce failed; see log)") + "\n"


def run_reduce(observations):
    return "\n".join(reduce_section(t, k, observations) for t, k in SECTIONS)


def replace_section(text, title, new_block):
    """Swap one '## title' block (up to the next '## ') in an existing model file."""
    m = re.search(rf"^## {re.escape(title)}\n.*?(?=^## |\Z)", text, flags=re.M | re.S)
    if not m:
        return text.rstrip("\n") + "\n\n" + new_block
    return text[:m.start()] + new_block + ("\n" if not new_block.endswith("\n\n") else "") + text[m.end():]


def derive_public(private_md):
    """Keep headings and [public] bullets only. The People section never crosses."""
    sections, cur = [], None
    for line in private_md.splitlines():
        if line.startswith("## "):
            cur = [line, []]
            sections.append(cur)
        elif cur and line.startswith("- ") and line.rstrip().endswith("[public]"):
            cur[1].append(TAG.sub("", line).rstrip())
    out = []
    for heading, bullets in sections:
        if heading == "## People":
            continue
        out.append(heading)
        out.append("")
        out.extend(bullets or ["- (nothing promoted yet)"])
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def header(label, observations, clips):
    dates = sorted({o["date"] for o in observations if o["date"] != "undated"})
    kinds = Counter(o["kind"] for o in observations)
    return (f"<!-- jason-model [{label}] — generated {time.strftime('%Y-%m-%d %H:%M')} by "
            f"persona/build-persona.py, model {MODEL}. {len(clips)} clips, {len(observations)} "
            f"observations ({', '.join(f'{k} {n}' for k, n in kinds.most_common())}). "
            f"Corpus spans {dates[0] if dates else '?'} → {dates[-1] if dates else '?'}. "
            f"{'PRIVATE. Never leaves this disk. Edit this file: flip [private]/[public], delete lines, then run --public.' if label == 'private' else 'Derived from the edited private file by --public. Do not edit; edit jason-model.md and re-derive.'} -->\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--reduce-only", action="store_true")
    ap.add_argument("--public", action="store_true", help="derive the public file from the edited private one")
    ap.add_argument("--limit", type=int, help="first N batches only")
    ap.add_argument("--only", metavar="SECTION", help="with --reduce-only: redo one section in place")
    args = ap.parse_args()

    clips = load_corpus()
    private_path = HERE / "jason-model.md"
    public_path = HERE / "jason-model.public.md"

    if args.public:
        if not private_path.is_file():
            sys.exit("no persona/jason-model.md to derive from")
        obs = load_observations()
        body = derive_public(private_path.read_text(encoding="utf-8"))
        public_path.write_text(header("public", obs, clips) + body + "\n", encoding="utf-8")
        n = sum(1 for l in body.splitlines() if l.startswith("- ") and "nothing promoted" not in l)
        print(f"wrote {public_path.name}: {n} public bullets")
        return

    if not args.reduce_only:
        run_map(clips, args.limit)
    if args.map_only:
        return
    obs = load_observations()
    if not obs:
        sys.exit("no observations in persona/extract/ — run the map first")
    if args.only:
        kinds = dict(SECTIONS).get(args.only)
        if not kinds:
            sys.exit(f"unknown section {args.only!r}; one of: {[t for t, _ in SECTIONS]}")
        block = reduce_section(args.only, kinds, obs)
        text = private_path.read_text(encoding="utf-8")
        private_path.write_text(replace_section(text, args.only, block), encoding="utf-8")
        print(f"replaced section {args.only!r} in {private_path.name}")
        return
    print(f"reduce: {len(obs)} observations, {len(SECTIONS)} sections, model {MODEL}")
    body = run_reduce(obs)
    private_path.write_text(header("private", obs, clips) + body, encoding="utf-8")
    print(f"wrote {private_path} ({len(body.split())} words). READ IT, edit it, then run --public.")


if __name__ == "__main__":
    main()
