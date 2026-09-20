#!/usr/bin/env python3
"""
ask.py — the Coach writes the questions for one episode.

    python3 prep/ask.py --topic "the pivot date" --out episodes/2026-09-20
    python3 prep/ask.py --topic "..." --n 8 --out episodes/<ep>
    python3 prep/ask.py --topic "..." --when 2025-10 --out episodes/<ep>   # only bullets dated that month
    python3 prep/ask.py --topic "..." --brief-only                           # print what the Coach would see

Reads coach/coach.md and model/jason-model.public.md from wcn-coach (COACH_REPO, or
COACH_MD / COACH_PUBLIC individually). The public file is the ONLY file about the guest
the Coach may see — a plain filter of the hand-edited private one; see wcn-coach's
model/build.py. Asks a local Ollama for N questions on the topic.

THE BRIEF
    The public file is ~25k words; the Coach gets a brief, not the file. Always: the
    timeline section ("What was going on, when") and the cadences. Then every bullet
    anywhere whose words overlap the topic, and — with --when — only bullets carrying a
    date in that month. Capped at BRIEF_WORDS. Deterministic, so the same topic gives the
    same brief; the model's variance is confined to the questions.

Writes <out>/questions.json:

    {"topic": ..., "model": ..., "generated": ...,
     "questions": [{"id": "q01", "text": "...", "rests_on": "...", "cites": "2025-09-28",
                    "audio": "q01.wav"}, ...]}

`rests_on` is the observation the question was built from, in the Coach's words, and
`cites` is its date — kept so a bad question can be traced to what it misread rather than
argued with. voice.py fills in the wavs; director.py plays them in order.

ONE QUESTION PER CALL
    Asked for twelve questions at once, the model wrote one and then eleven variations
    of it — the same closing formula, the same quotes recycled, the topic bolted onto
    every tail. Rules and retries did not fix it, because a model completing a list is
    pattern-matching its own previous line. So the SCRIPT picks the observations — spread
    across sections and dates, skipping garbled transcript lines — and the model writes
    ONE question per observation, never seeing the others. Then a validator checks the
    set for tics and re-asks the offenders with a "not like this" note.

THE SPOT-CHECK IS THE TEST
    Print the questions and read them. If one could be asked of anyone, the persona is
    not done — fix the file, not the prompt. This is the same rule the Coach's Note
    already lives under: if it could apply to any participant, rewrite it.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The persona lives in wcn-coach (private) since 2026-09-20. Booth reads ONLY the public
# derivation of it, and the Coach's register, from there.
COACH_REPO = Path(os.environ.get("COACH_REPO", Path.home() / "Projects" / "wcn-coach"))
COACH = Path(os.environ.get("COACH_MD", COACH_REPO / "coach" / "coach.md"))
PUBLIC = Path(os.environ.get("COACH_PUBLIC", COACH_REPO / "model" / "jason-model.public.md"))
OLLAMA = os.environ.get("BOOTH_OLLAMA", "http://192.168.1.118:11434")
MODEL = os.environ.get("BOOTH_MODEL", "qwen3:30b")
BRIEF_WORDS = 8000
# "(dates)" and, from wcn-coach's builder, an optional " · context, context" after them
TAGLESS = re.compile(r"\s*\((20\d\d-\d\d-\d\d(?:, 20\d\d-\d\d-\d\d)*)\)(?:\s*·[^\n]*)?\s*$")
ALWAYS = ("What was going on, when", "Named cadences and rituals")
STOP = set("the a an and or of to in on at for with about from by is was were be been it its this that "
           "these those they them their he she his her we our you your i my me what when where how why "
           "which who whom not no yes do does did done have has had will would could should can may".split())

ONE = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


def ollama_chat(system, user, schema, temperature=0.7, timeout=900, tries=3):
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False, "think": False, "format": schema,
        "options": {"temperature": temperature, "num_ctx": 16384, "num_predict": 400},
    }
    for attempt in range(tries):
        req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content = json.loads(r.read())["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Ran on inside a string until num_predict. Warm it up and go again.
            body["options"]["temperature"] = min(1.0, temperature + 0.1 * (attempt + 1))
    raise SystemExit(f"model returned unusable JSON {tries} times: {content[:120]!r}")


def sections(md):
    """{heading: [bullet lines]} from a model file."""
    out, cur = {}, None
    for line in md.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip(); out[cur] = []
        elif cur and line.startswith("- "):
            out[cur].append(line)
    return out


def brief(public_md, topic, when=None):
    secs = sections(public_md)
    words = {w for w in re.findall(r"[a-z][a-z'-]{2,}", topic.lower()) if w not in STOP}
    chosen, total = [], 0

    def take(heading, lines):
        nonlocal total
        keep = [l for l in lines if not when or when in l]
        if not keep:
            return
        block = f"## {heading}\n" + "\n".join(keep)
        n = len(block.split())
        if total + n > BRIEF_WORDS and heading not in ALWAYS:
            keep = keep[: max(1, int(len(keep) * (BRIEF_WORDS - total) / n))]
            block = f"## {heading}\n" + "\n".join(keep)
            n = len(block.split())
        chosen.append(block); total += n

    for h in ALWAYS:
        if h in secs:
            take(h, secs[h])
    for h, lines in secs.items():
        if h in ALWAYS:
            continue
        hits = [l for l in lines if words & set(re.findall(r"[a-z][a-z'-]{2,}", l.lower()))]
        if hits:
            take(h, hits)
        if total >= BRIEF_WORDS:
            break
    return "\n\n".join(chosen), total


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def spoken_dates(text):
    """'On the 2025-10-02' → 'On October 2nd, 2025'. The TTS reads what is written."""
    def sub(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        suf = "th" if 11 <= d % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
        return f"{MONTHS[mo-1]} {d}{suf}, {y}"
    text = re.sub(r"\b(?:on|On) the (20\d\d)-(\d\d)-(\d\d)", lambda m: ("On " if m.group(0)[0] == "O" else "on ") + sub(m), text)
    return re.sub(r"\b(20\d\d)-(\d\d)-(\d\d)\b", sub, text)


def validate(qs):
    """Reject the two ways a model pads a list: reused quotes and a repeated closing formula."""
    problems, seen_quotes, closers = [], {}, {}
    for i, q in enumerate(qs, 1):
        q["text"] = re.sub(r"\s*\n\s*", " ", q["text"]).strip()
        if not q["text"].rstrip().endswith("?"):
            problems.append(f"q{i} is not a question")
        for quote in re.findall(r"'([^']{12,})'", q["text"]):
            key = quote[:40].lower()
            if key in seen_quotes:
                problems.append(f"q{i} reuses the quote '{quote[:30]}…' from q{seen_quotes[key]}")
            seen_quotes.setdefault(key, i)
        tail = re.sub(r"[^a-z ]", "", q["text"].strip().lower().split(".")[-1].split("?")[0])[-40:]
        if tail in closers:
            problems.append(f"q{i} ends the same way as q{closers[tail]}")
        closers.setdefault(tail, i)
    # A tic: the same two words opening or closing the QUESTION SENTENCE on three or more.
    ends, opens = {}, {}
    for i, q in enumerate(qs, 1):
        sent = [x for x in re.split(r"(?<=[.!?])\s+", q["text"]) if x.strip()]
        qsent = next((x for x in reversed(sent) if x.strip().endswith("?")), sent[-1] if sent else "")
        w = re.findall(r"[a-z']+", qsent.lower())
        if len(w) >= 2:
            ends.setdefault(" ".join(w[-2:]), []).append(i)
            opens.setdefault(" ".join(w[:2]), []).append(i)
    for label, d in (("closes", ends), ("opens", opens)):
        for phrase, qi in d.items():
            if len(qi) >= 3:
                problems.append(f"'{phrase}' {label} {len(qi)} questions (q{', q'.join(map(str, qi))}); vary them")
    return qs, problems


def usable(line):
    """Skip bullets whose quote is transcript mush: ellipses, no punctuation, or very long."""
    q = re.search(r"'([^']+)'", line)
    if not q:
        return True
    t = q.group(1)
    return "..." not in t and len(t.split()) <= 45 and not re.search(r"\b(\w+)( \1){3,}\b", t.lower())


def pick(brief_md, n, when=None):
    """n bullets, round-robin across sections (timeline first), each from a different date
    where possible, in date order. Deterministic."""
    secs = sections(brief_md)
    order = [h for h in ("What was going on, when",) if h in secs] + [h for h in secs if h != "What was going on, when"]
    pools = {h: [l for l in secs[h] if usable(l) and (not when or when in l)] for h in order}
    chosen, used_dates = [], set()
    while len(chosen) < n and any(pools.values()):
        for h in order:
            if len(chosen) >= n or not pools[h]:
                continue
            # prefer a bullet whose date we have not used yet
            idx = next((i for i, l in enumerate(pools[h])
                        if not (set(re.findall(r"20\d\d-\d\d-\d\d", l)) & used_dates)), 0)
            line = pools[h].pop(idx)
            used_dates |= set(re.findall(r"20\d\d-\d\d-\d\d", line))
            chosen.append((h, line))
    def first_date(item):
        d = re.findall(r"20\d\d-\d\d-\d\d", item[1])
        return min(d) if d else "9999"
    return sorted(chosen, key=first_date)


def ask_one(system, topic, section, line, avoid=None, temperature=0.8):
    user = (f"The episode is about: {topic}. That is context only — do not mention it or tie the "
            f"question back to it.\n\nWrite ONE question, following 'How a question is built', resting "
            f"on this single observation from the '{section}' section of the file:\n\n{line}\n\n"
            "Name the date it carries, quote at most twenty of their words (the clearest clause), "
            "one sentence of setup, one question, under forty words spoken aloud.")
    if avoid:
        user += ("\n\nOther questions in this episode already open or close like these — yours must "
                 "open AND close differently from all of them: " + " | ".join(f'"{a}"' for a in avoid))
    out = ollama_chat(system, user, ONE, temperature=temperature)
    return out["text"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, help="what this episode is about")
    ap.add_argument("--when", help="YYYY-MM: only bullets carrying a date in that month")
    ap.add_argument("--brief-only", action="store_true", help="print the brief and exit")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", help="episode directory")
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    if not PUBLIC.is_file():
        sys.exit(f"{PUBLIC} does not exist — build it in wcn-coach (model/build.py, edit, --public)")
    coach = COACH.read_text(encoding="utf-8")
    public, n_words = brief(PUBLIC.read_text(encoding="utf-8"), args.topic, args.when)
    if args.brief_only:
        print(public); print(f"\n[{n_words} words]", file=sys.stderr); return
    if not args.out:
        sys.exit("--out is required (or --brief-only)")
    print(f"brief: {n_words} words for topic {args.topic!r}" + (f", month {args.when}" if args.when else ""))

    # The Coach sees the timeline for context, and one observation per call.
    timeline = sections(public).get("What was going on, when", [])
    system = coach + "\n\n# The guest's timeline (context)\n\n" + "\n".join(timeline)

    t0 = time.time()
    picked = pick(public, args.n, args.when)
    print(f"picked {len(picked)} observations across {len({h for h, _ in picked})} sections")
    qs = []
    for section, line in picked:
        text = ask_one(system, args.topic, section, line, temperature=args.temperature)
        cites = ", ".join(sorted(set(re.findall(r"20\d\d-\d\d-\d\d", line))))
        qs.append({"text": text, "rests_on": TAGLESS.sub("", line[2:]).strip(), "cites": cites, "section": section})
        print(f"  · {text[:90]}")
    # Tics across the set: re-ask the offenders, telling them what not to sound like.
    for round_ in range(3):
        qs, problems = validate(qs)
        if not problems:
            break
        bad = sorted({int(m) for pr in problems for m in re.findall(r"q(\d+)", pr)})
        print(f"  tics: {len(problems)} — re-asking q{', q'.join(map(str, bad))}")
        for i in bad:
            sec, line = picked[i - 1]
            avoid = [q["text"].split(".")[-1].strip()[:60] for j, q in enumerate(qs, 1) if j != i][:8]
            qs[i - 1]["text"] = ask_one(system, args.topic, sec, line, avoid=avoid, temperature=0.9)
    dropped = [q for q in qs if not q["text"].rstrip().endswith("?")]
    qs = [q for q in qs if q["text"].rstrip().endswith("?")]
    for q in dropped:
        print(f"  dropped (not a question): {q['text'][:70]}")
    for i, q in enumerate(qs, 1):
        q["id"] = f"q{i:02d}"
        q["audio"] = f"q{i:02d}.wav"
        q["text"] = spoken_dates(q["text"].strip())

    ep = Path(args.out)
    ep.mkdir(parents=True, exist_ok=True)
    doc = {"topic": args.topic, "model": MODEL, "generated": time.strftime("%Y-%m-%dT%H:%M"),
           "questions": qs}
    (ep / "questions.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(qs)} questions in {time.time()-t0:.0f}s → {ep/'questions.json'}\n")
    for q in qs:
        print(f"{q['id']}  {q['text']}")
        print(f"      ↳ {q['rests_on']}  ({q['cites']})\n")
    print("Spot-check: could any of these be asked of anyone? If yes, fix the file, not the prompt.")


if __name__ == "__main__":
    main()
