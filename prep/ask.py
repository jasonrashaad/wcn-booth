#!/usr/bin/env python3
"""
ask.py — the Coach writes the questions for one episode.

    python3 prep/ask.py --topic "the pivot date" --out episodes/2026-09-20
    python3 prep/ask.py --topic "..." --n 8 --out episodes/<ep>

Reads persona/coach.md (the register and the rules) and persona/jason-model.public.md
(the ONLY file about the guest the Coach may see — hand-edited, see build-persona.py),
and asks a local Ollama for N questions on the topic. Writes <out>/questions.json:

    {"topic": ..., "model": ..., "generated": ...,
     "questions": [{"id": "q01", "text": "...", "rests_on": "...", "cites": "2025-09-28",
                    "audio": "q01.wav"}, ...]}

`rests_on` is the observation the question was built from, in the Coach's words, and
`cites` is its date — kept so a bad question can be traced to what it misread rather than
argued with. voice.py fills in the wavs; director.py plays them in order.

THE SPOT-CHECK IS THE TEST
    Print the questions and read them. If one could be asked of anyone, the persona is
    not done — fix the file, not the prompt. This is the same rule the Coach's Note
    already lives under: if it could apply to any participant, rewrite it.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COACH = REPO / "persona" / "coach.md"
PUBLIC = REPO / "persona" / "jason-model.public.md"
OLLAMA = os.environ.get("BOOTH_OLLAMA", "http://192.168.1.118:11434")
MODEL = os.environ.get("BOOTH_MODEL", "qwen3:30b")

SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "rests_on": {"type": "string"},
                    "cites": {"type": "string"},
                },
                "required": ["text", "rests_on", "cites"],
            },
        }
    },
    "required": ["questions"],
}


def ollama_chat(system, user, schema, temperature=0.7, timeout=900):
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False, "think": False, "format": schema,
        "options": {"temperature": temperature, "num_ctx": 16384},
    }
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(json.loads(r.read())["message"]["content"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, help="what this episode is about")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", required=True, help="episode directory")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    if not PUBLIC.is_file():
        sys.exit(f"{PUBLIC} does not exist — run persona/build-persona.py, then READ AND EDIT it first")
    coach = COACH.read_text(encoding="utf-8")
    public = PUBLIC.read_text(encoding="utf-8")

    system = coach + "\n\n# The guest's file (public)\n\n" + public
    user = (f"Write {args.n} questions for an episode about: {args.topic}.\n\n"
            "Each question follows 'How a question is built'. For each, also give `rests_on`: the "
            "one observation from the file it is built from, in one sentence, and `cites`: that "
            "observation's date(s) exactly as they appear in the file. Spread the questions across "
            "different observations; at least two should be built from the Tensions section if it "
            "has anything. Order them the way you would ask them.")

    t0 = time.time()
    out = ollama_chat(system, user, SCHEMA, temperature=args.temperature)
    qs = out["questions"][: args.n]
    for i, q in enumerate(qs, 1):
        q["id"] = f"q{i:02d}"
        q["audio"] = f"q{i:02d}.wav"
        q["text"] = q["text"].strip()

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
