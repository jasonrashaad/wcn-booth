#!/usr/bin/env python3
"""
themes.py — what they were talking about, and when. A word cloud with dates.

    .venv/bin/python persona/themes.py                        # write persona/themes.md
    .venv/bin/python persona/themes.py --with treadmill       # when did it come up, and what else was in the room?
    .venv/bin/python persona/themes.py --with "gas station" --with costco
    .venv/bin/python persona/themes.py --append       # add the section to persona/jason-model.md, all [private]

No LLM. Deterministic, so the same corpus gives the same answer every time — which a
derived taxonomy does not (wcn-transcript, findings §6b). Needs the .venv for spaCy
(`en_core_web_sm`, already there as a Kokoro dependency): a word cloud is a cloud of
NOUNS, and telling "treadmill" from "somewhere" is a part-of-speech question, not a
frequency one — counts alone kept ranking texture words, because the weeks are too
unequal (15 vs 161 entries) for burstiness to be significant.

The corpus is journal entries, not goal statements. The question a coach asks of a
journal is not "what did you commit to" but "what was on your mind, and when, and what
travelled with it". Three views answer that:

  CLOUD      noun lemmas (and noun-noun compounds, so "gas station" survives as a
             thing) ranked by how many entries they appear in, with the week each one
             peaks in. Proper nouns are kept: places and products are topics; people
             are too, and the People section of the model is where they get fenced.
  BY WEEK    for each ISO week, the terms most over-represented in that week against the
             whole corpus (log-ratio of entry shares, with a floor so one mention cannot
             win). "What was that month about?"
  WITH X     entries mentioning X: their dates by week, and the terms over-represented in
             those entries against the rest. "What else was I talking about when I talked
             about that?"

Every number here is a count of entries (clips), never of words — an entry that says
"treadmill" nine times is one entry that was about the treadmill.

Input: whatever build-persona.load_corpus() loads — transcript JSON today, dated
Markdown (Evolution records) when the loader is pointed at them. Output is PRIVATE and
gitignored, like everything else derived from the corpus.
"""
import argparse
import collections
import datetime
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
bp = __import__("build-persona")

# Nouns that are texture, not topic, in spoken English — plus the profanity spaCy tags
# as nouns, and the speaker's own name.
HAND_STOP = set("""
thing things stuff something anything nothing everything someone anyone everyone
nobody anybody everybody kind sort lot lots bit way ways time times day days year years
week weeks minute minutes second seconds hour hours today tomorrow yesterday tonight
guy guys dude man men people person point part end start reason problem idea question
shit fuck fucking motherfucker motherfucking bitch ass damn hell crap goddamn god
bullshit asshole pussy dick nigga bro
fact one bunch sense
jason okay yeah
""".split())
CORPUS_STOP_N = 0          # unused since the noun filter; kept so the knob still exists
MIN_ENTRIES = 3            # a term must appear in at least this many entries to rank anywhere
MIN_WEEK = 5               # weeks with fewer entries than this cannot make a term "peak"
PEAK = 2.0                 # a topic's best week must beat its overall share by this much
TOP = 24


_nlp = None


def nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return _nlp


def tokens(text):
    """Noun lemmas + adjacent noun-noun compounds, as a set (entry-level presence)."""
    doc = nlp()(text)
    out, prev = set(), None
    for tok in doc:
        ok = tok.pos_ in ("NOUN", "PROPN") and tok.is_alpha and len(tok) >= 3
        lemma = tok.lemma_.lower() if ok else None
        if lemma == "datum":      # spaCy singularises "data"; nobody says datum
            lemma = "data"
        if ok and lemma not in HAND_STOP:
            out.add(lemma)
            if prev:
                out.add(f"{prev} {lemma}")
        prev = lemma if ok and lemma not in HAND_STOP else None
    return out


def week_of(date):
    y, w, _ = datetime.date.fromisoformat(date).isocalendar()
    return f"{y}-W{w:02d}"


def load():
    entries = []
    for c in bp.load_corpus():
        if c["date"] == "undated":
            continue
        entries.append({"id": c["id"], "date": c["date"], "week": week_of(c["date"]),
                        "terms": tokens(c["text"])})
    df = collections.Counter(t for e in entries for t in e["terms"])
    return entries, df


def over_represented(subset, entries, df, top=TOP):
    """Terms whose share of `subset` entries beats their share of all entries."""
    n_sub, n_all = len(subset), len(entries)
    sub_df = collections.Counter(t for e in subset for t in e["terms"])
    scored = []
    for t, k in sub_df.items():
        if k < MIN_ENTRIES:
            continue
        share_sub = k / n_sub
        share_all = df[t] / n_all
        scored.append((math.log(share_sub / share_all), k, t))
    scored.sort(reverse=True)
    return [(t, k, round(s, 2)) for s, k, t in scored[:top]]


def cloud(entries, df, top=80):
    """Nouns by entry count, each with the week it peaks in."""
    n_all = len(entries)
    weeks = collections.defaultdict(list)
    for e in entries:
        weeks[e["week"]].append(e)
    big = {wk: es for wk, es in weeks.items() if len(es) >= MIN_WEEK}
    wdf = {wk: collections.Counter(t for e in es for t in e["terms"]) for wk, es in big.items()}
    out = []
    for t, k in df.most_common():
        if k < MIN_ENTRIES:
            break
        share_all = k / n_all
        peak = max((wdf[wk][t] / len(big[wk])) / share_all for wk in big) if big else 0
        best = max(big, key=lambda wk: wdf[wk][t] / len(big[wk])) if big else "?"
        out.append((t, k, best, round(peak, 1)))
        if len(out) >= top:
            break
    return out


def by_week(entries, df):
    weeks = collections.defaultdict(list)
    for e in entries:
        weeks[e["week"]].append(e)
    return [(wk, len(es), over_represented(es, entries, df, top=14)) for wk, es in sorted(weeks.items())]


def with_term(term, entries, df):
    term = term.lower().strip()
    hits = [e for e in entries if term in e["terms"]
            or any(term == t or term in t.split() for t in e["terms"])]
    weeks = collections.Counter(e["week"] for e in hits)
    total_by_week = collections.Counter(e["week"] for e in entries)
    return hits, weeks, total_by_week, over_represented(hits, entries, df) if hits else []


def render(entries, df):
    out = [f"<!-- themes — generated {datetime.date.today()} by persona/themes.py from "
           f"{len(entries)} dated entries. PRIVATE: derived from the journal. -->", "",
           "# What they were talking about, and when", "",
           f"{len(entries)} dated entries, {entries[0]['date']} → {entries[-1]['date']}. "
           "Every number is a count of entries, not words.", "",
           "## The cloud — topical terms by how many entries they appear in", ""]
    for t, k, best, peak in cloud(entries, df):
        out.append(f"- **{t}** — {k} entries, peaks {best} ({peak}× its overall share)")
    out += ["", "## By week — what was distinctive about that week", ""]
    for wk, n, terms in by_week(entries, df):
        out.append(f"### {wk} · {n} entries")
        out.append("")
        out.append(", ".join(f"{t} ({k})" for t, k, _ in terms) or "(too few entries to rank)")
        out.append("")
    out += ["## How to ask the next question", "",
            "`python3 persona/themes.py --with <term>` — when it came up, and what travelled with it.", ""]
    return "\n".join(out)


SECTION = "## What they were talking about, and when"


def model_section(entries, df):
    """The cloud and the weeks as tagged bullets for jason-model.md. All [private]:
    a topic is promoted by the person, never by a script."""
    lines = [SECTION, "", "<!-- from persona/themes.py — counts of entries, not words. "
             "Re-run `themes.py --append` to refresh; it replaces this section only. -->", ""]
    for t, k, best, peak in cloud(entries, df, top=40):
        lines.append(f"- **{t}**: {k} entries, peaking {best} ({peak}× its overall share) [private]")
    for wk, n, terms in by_week(entries, df):
        if n < MIN_WEEK:
            continue
        lines.append(f"- {wk} ({n} entries) was about: "
                     + ", ".join(f"{t} ({k})" for t, k, _ in terms[:10]) + f" ({wk}) [private]")
    return "\n".join(lines) + "\n"


def append_to_model(entries, df):
    path = HERE / "jason-model.md"
    if not path.is_file():
        sys.exit("no persona/jason-model.md yet — run build-persona.py first")
    text = path.read_text(encoding="utf-8")
    section = model_section(entries, df)
    if SECTION in text:
        head, _, rest = text.partition(SECTION)
        # drop the old section up to the next heading (or EOF)
        nxt = re.search(r"^## ", rest, flags=re.M)
        text = head + section + ("\n" + rest[nxt.start():] if nxt else "")
    else:
        text = text.rstrip("\n") + "\n\n" + section
    path.write_text(text, encoding="utf-8")
    print(f"{path.name}: section '{SECTION[3:]}' written ({section.count(chr(10)) - 4} bullets, all [private])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with", dest="terms", action="append", metavar="TERM",
                    help="query a term (repeatable) instead of writing themes.md")
    ap.add_argument("--append", action="store_true", help="write the section into persona/jason-model.md")
    args = ap.parse_args()
    entries, df = load()

    if args.append:
        append_to_model(entries, df)
        return

    if args.terms:
        for term in args.terms:
            hits, weeks, totals, terms = with_term(term, entries, df)
            print(f"\n== {term!r}: {len(hits)} of {len(entries)} entries")
            if not hits:
                continue
            print("   by week:", "  ".join(f"{wk} {weeks[wk]}/{totals[wk]}" for wk in sorted(totals) if weeks[wk]))
            print("   travels with:", ", ".join(f"{t} ({k})" for t, k, _ in terms[:16]) or "(nothing distinctive)")
            print("   entries:", ", ".join(sorted(e["date"] for e in hits)[:12]), "…" if len(hits) > 12 else "")
        return

    out = HERE / "themes.md"
    out.write_text(render(entries, df), encoding="utf-8")
    print(f"wrote {out} — {len(entries)} entries, {len(df)} terms")


if __name__ == "__main__":
    main()
