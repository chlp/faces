# faces

Face recognition app for Orange Pi 5 Max (RK3588): a camera watches the entrance, recognizes people by face and greets them by name via espeak-ng. Detection and recognition run on the NPU (SCRFD + ArcFace models in RKNN format).

## What's inside

- `main.py` — main application (entry point, core logic)
- `known_faces/` — face database: one folder per person, containing photos (`.jpg` / `.png`)
- `models/` — RKNN models (downloaded via `download_models.sh`)
- `data/` — runtime data (auto-created)
  - `faces.db` — SQLite: event log + snapshots (BLOBs), auto-pruned to 15 entries
- `install.sh` — installs system dependencies and Python environment on Orange Pi
- `download_models.sh` — downloads SCRFD and ArcFace models from rknn_model_zoo
- `requirements.txt` — Python packages (opencv, numpy, Pillow, etc.)

## Quick start

```bash
# 1. Install dependencies (system packages + venv + pip)
chmod +x install.sh && ./install.sh

# 2. RKNN runtime is not on PyPI — download the .whl manually and place it in the project directory:
#    https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages
#    (pick the file for your architecture, e.g. linux_aarch64)

# 3. Download detection and recognition models
./download_models.sh

# 4. Add yourself to the database
mkdir -p known_faces/Alice
cp ~/photo.jpg known_faces/Alice/photo.jpg

# 5. Run
source venv/bin/activate
python3 main.py
```

## CLI arguments

```bash
python3 main.py                                    # all defaults
python3 main.py --port 9090 --threshold 0.50       # custom port and threshold
python3 main.py --lang en --no-tts                  # English, no TTS
python3 main.py --camera 2 --data-dir /mnt/data     # specific camera, custom data dir
python3 main.py --no-web --display                   # no web server, OpenCV window
python3 main.py --no-debug                           # no debug output to console
```

Environment variables (set defaults; CLI takes priority):
`FACE_PORT`, `FACE_THRESHOLD`, `FACE_LANG`, `FACE_CAMERA`, `FACE_DATA_DIR`

## Web interface

Open in a browser: `http://<orange-pi-ip>:8080`

Shows a live camera feed (~3 fps) and the last 15 events with thumbnails. Frames are served from memory (no disk writes). The `↻` button reloads the face database without restarting.

### Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Web UI |
| `GET /frame.jpg` | Live JPEG frame (from memory) |
| `GET /detections.json` | Recent events (JSON) |
| `GET /snap/<id>.jpg` | Event snapshot (BLOB from SQLite) |
| `GET /health` | Status: `{uptime_s, last_detection_ts, frame_jpeg_bytes}` |
| `GET /reload` | Reload face database (hot-reload) |
| `GET /clear` | Delete all events and snapshots from DB |
| `GET /debug/aligned.jpg` | Last aligned face (debug) |

## Adding new people

Create a folder with the person's name and put one or more photos inside (face clearly visible):

```
known_faces/
  Alice/
    photo.jpg
  Bob/
    photo.jpg
    photo2.jpg
```

The database reloads automatically every 30 seconds when files change, or via the `↻` button in the web UI, or through `GET /reload`.

## Settings

| Parameter | Default | CLI | Description |
|-----------|---------|-----|-------------|
| `recognition_threshold` | 0.45 | `--threshold` | Cosine similarity cutoff (higher = stricter) |
| `stranger_min_score` | 0.30 | — | Below this — not counted as a stranger (filters false positives) |
| `greet_cooldown` | 10 s | — | Pause between greetings for the same person |
| `confirm_frames` | 3 | — | Consecutive frames required for confirmation |
| `score_window` | 7 | — | Score smoothing window |
| `web_event_cooldown` | 30 s | — | Minimum interval between identical events |
| `stranger_confirm_delay` | 5 s | — | Delay before recording a stranger |
| `lang` | ru | `--lang` | Greeting language (ru / en) |
| `web_port` | 8080 | `--port` | Web interface port (0 = disabled) |

## Autostart on boot

On Orange Pi under the `orangepi` user:

```bash
# 1. Copy the unit to user systemd
mkdir -p ~/.config/systemd/user
cp /path/to/faces/faces.service ~/.config/systemd/user/

# 2. Enable lingering (once!)
sudo loginctl enable-linger orangepi

# 3. Enable and start the service
systemctl --user daemon-reload
systemctl --user enable faces
systemctl --user start faces
```

Useful commands:

- **Logs**: `tail -f ~/faces/faces.log`
- **Restart**: `systemctl --user restart faces`
- **Stop**: `systemctl --user stop faces`
- **Disable autostart**: `systemctl --user disable faces`

### If the service doesn't start after reboot

1. **Linger**: `loginctl show-user $USER | grep Linger` — should be `Linger=yes`
2. **Enabled**: `systemctl --user is-enabled faces` — should be `enabled`
3. **Status**: `systemctl --user status faces` + `tail -n 80 ~/faces/faces.log`
