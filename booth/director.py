#!/usr/bin/env python3
"""
director.py — serve the avatar page and one episode's questions to OBS.

    python3 booth/director.py episodes/<ep>          # http://127.0.0.1:8788/

Stdlib only, loopback only. OBS loads http://127.0.0.1:8788/ as a Browser Source
(1280x720, "Control audio via OBS" on). State is one integer: the current question.
    GET /api/state        -> {"index": n, "count": N, "question": {...} | null}
    GET /api/next         -> advance (or wrap to 0) and return the new state
    GET /api/goto?i=n     -> jump
    GET /ep/<file>        -> that episode's wavs / timelines
The page polls /api/state; when the index changes it plays q<NN>.wav and drives
the mouth off q<NN>.timeline.json. Bind /api/next to a hotkey with a one-line curl.

Same shape as wcn-commandcenter's state server on :8787 — deliberately.
"""
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
PORT = 8788


class State:
    def __init__(self, episode: Path):
        self.episode = episode
        qfile = episode / "questions.json"
        self.questions = json.loads(qfile.read_text())["questions"] if qfile.is_file() else []
        self.index = -1        # nothing playing yet
        self.serial = 0        # bumps on every trigger so a repeat of the same index replays

    def view(self):
        q = self.questions[self.index] if 0 <= self.index < len(self.questions) else None
        return {"index": self.index, "serial": self.serial,
                "count": len(self.questions), "question": q}


class Handler(SimpleHTTPRequestHandler):
    state: State = None

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/state":
            return self.json(self.state.view())
        if u.path == "/api/next":
            self.state.index = (self.state.index + 1) % max(1, len(self.state.questions))
            self.state.serial += 1
            return self.json(self.state.view())
        if u.path == "/api/goto":
            self.state.index = int(parse_qs(u.query).get("i", ["0"])[0])
            self.state.serial += 1
            return self.json(self.state.view())
        self.route(u)
        return super().do_GET()

    def do_HEAD(self):
        self.route(urlparse(self.path))
        return super().do_HEAD()

    def route(self, u):
        """/ep/* is the episode dir; everything else is this directory."""
        if u.path.startswith("/ep/"):
            self.path = "/" + u.path[len("/ep/"):]
            self.directory = str(self.state.episode)
        else:
            # SimpleHTTP only auto-serves index.html; "/" must map to the avatar page
            # explicitly or OBS gets a directory listing (which is what happened).
            if u.path == "/":
                self.path = "/avatar.html"
            self.directory = str(HERE)

    def json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # Never let OBS's embedded browser cache an old episode.
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "/api/state" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    Handler.state = State(Path(sys.argv[1]).resolve())
    print(f"episode {Handler.state.episode.name}: {len(Handler.state.questions)} question(s)")
    print(f"http://127.0.0.1:{PORT}/   next: curl -s http://127.0.0.1:{PORT}/api/next")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
