# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal face recognition app for Orange Pi 5 Max (ARM64, Ubuntu/Debian). A USB webcam watches the apartment entrance; when a known person appears, the app greets them by name via espeak-ng TTS.

## Hardware target

- **Board**: Orange Pi 5 Max (RK3588, ARM64)
- **Camera**: Anker PowerConf C200 — USB, shows up as `/dev/video0`
- **OS**: Ubuntu/Debian ARM64

## Setup & run

```bash
chmod +x install.sh && ./install.sh   # one-time: installs deps, compiles dlib (~15 min)
source venv/bin/activate
python3 main.py
```

## Adding people to the database

```bash
mkdir -p known_faces/NAME
# place one or more photos as known_faces/NAME/photo.jpg
```

Multiple photos per person improve accuracy. Face must be clearly visible.

## Key architecture

- `main.py` — single-file app, all logic here
- `known_faces/` — face database, one subfolder per person
- Recognition runs on every `PROCESS_EVERY_N`-th frame at `FRAME_SCALE` resolution to save CPU
- Greeting cooldown (`GREET_COOLDOWN` seconds) prevents repeated greetings
- TTS runs in a daemon thread via `subprocess` → `espeak-ng`

## Tuning knobs (top of main.py)

| Variable | Default | Effect |
|---|---|---|
| `CAMERA_INDEX` | 0 | Try 1 or 2 if camera not found |
| `FRAME_SCALE` | 0.5 | Lower = faster but less accurate |
| `RECOGNITION_TOLERANCE` | 0.55 | Lower = stricter matching |
| `GREET_COOLDOWN` | 300 | Seconds between greetings per person |
| `PROCESS_EVERY_N` | 3 | Process every Nth frame |
| `LANG` | "ru" | "ru" or "en" |
