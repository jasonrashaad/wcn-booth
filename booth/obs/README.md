# OBS — the WCN Booth profile and scene collection

Reference copies of what lives in `~/Library/Application Support/obs-studio/basic/`
on wcn-macbook (built 2026-09-18). OBS owns the live files; these are for rebuilding.

- `WCN_Booth.json` → `basic/scenes/` — scene collection **WCN Booth**, one scene `Booth`:
  `Set` (colour placeholder for the brand set) · `Coach` (browser source,
  `http://127.0.0.1:8788/`, 1280×720, audio via OBS, **Monitor and Output**, tracks 1+2)
  · `Camera` (Logitech StreamCam, right half) · `Mic` (built-in, tracks 1+3).
- `WCN_Booth.basic.ini` → `basic/profiles/WCN_Booth/basic.ini` — profile **WCN Booth**:
  1920×1080 @ 30, hybrid mp4 (fragmented — survives a crash, ffmpeg reads it directly),
  records to `~/Movies/booth/`, **three audio tracks**: 1 = full mix, 2 = Coach only,
  3 = mic only. Track 2 is what post uses for the clean question audio.

The pre-existing profile was renamed **WCN Sparks** (was `Untitled`) and left as it was,
including its recording path. Sparks records vertical; the booth is 16:9. Switch with
Profile ▸ and Scene Collection ▸ in the OBS menu bar — they are independent.

To restore: quit OBS, copy the two files into place, set `Profile`/`ProfileDir`/
`SceneCollection`/`SceneCollectionFile` in `user.ini` `[Basic]`, relaunch.
