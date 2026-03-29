# Gesture Game

A two-player real-time gesture battle game controlled entirely by hand gestures from a single webcam. Each round both players simultaneously show **Fist (Attack)**, **Open Hand (Defend)**, or **Peace Sign (Heal)**. The system locks both inputs, applies battle rules, and updates HP until one player reaches 0.

---

## Setup

### 1. Create and activate a Python environment (recommended: venv, Python 3.11)

```bash
cd /path/to/Gesture-Game-main
python3.11 -m venv .venv
source .venv/bin/activate
```

> Conda is also fine if you prefer it. The game has been tested with Python 3.11 + mediapipe 0.10.32.

### 2. Install Python packages

```bash
pip install -r requirements.txt
```

### 3. (macOS) Install ffmpeg (recommended)

`WebcamLoader` uses ffmpeg for low-latency capture when available.
If ffmpeg is missing, the code automatically falls back to OpenCV camera capture.

```bash
brew install ffmpeg
```

### 4. Camera permission (macOS)

Allow camera access for your terminal/IDE:
- System Settings → Privacy & Security → Camera

### 5. MediaPipe model file

`gesture_game/processor/hand_landmarker.task` is already included in this repo.
If missing, it is auto-downloaded on first run.

### 6. Run the game

```bash
cd gesture_game
python main.py
```

### Optional: choose a specific camera

```bash
# by camera display name (ffmpeg path)
GESTURE_CAMERA_NAME="FaceTime HD Camera" python main.py

# by numeric camera index (OpenCV fallback path)
GESTURE_CAMERA_INDEX=1 python main.py
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | 4.10.0.84 | Webcam capture, frame drawing, display window |
| `mediapipe` | 0.10.32 | 21-point hand landmark detection (CPU, real-time) |
| `numpy` | 1.26.4 | Frame slicing and landmark array ops |
| `ffmpeg` (system) | 8.x+ recommended | Low-latency camera capture + camera selection by name |

> **Note:** numpy is pinned below 2.0 because mediapipe 0.10.x requires it.

---

## Gestures

| Gesture | Hand shape | Action |
|---|---|---|
| Attack | Fist (all fingers curled) | Deals 1 HP damage |
| Defend | Open palm (all fingers extended) | Blocks incoming attack |
| Heal | Peace sign (index + middle up) | Restores 1 HP (capped at 3) |

## Rules

- **Attack vs Attack** → both lose 1 HP
- **Attack vs Defend** → blocked, no damage
- **Attack vs Heal** → attacker hits; healer loses 1 HP
- **Defend vs Defend** → stalemate
- **Heal vs Heal** → both gain 1 HP

## Controls

| Key | Action |
|---|---|
| `Q` | Quit |
| `R` | Restart (on game-over screen) |

## Troubleshooting

- `Could not open camera` or `camera access has been denied` on macOS
  - Enable camera permission for Terminal / VS Code in System Settings.
  - Try a different device index: `GESTURE_CAMERA_INDEX=1 python main.py`
- `Camera 'FaceTime HD Camera' not found`
  - Use another name via `GESTURE_CAMERA_NAME="..."` or use `GESTURE_CAMERA_INDEX`.

---

## Git Workflow

> **Never commit directly to `main`.** Always create a new branch for your work.

### First-time clone
```bash
git clone https://github.com/IvanJ530/Gesture-Game.git
cd Gesture-Game
```

### Every time you start working — create a new branch
```bash
# Create and switch to a new branch (use a descriptive name)
git checkout -b feature/your-feature-name

# Examples:
git checkout -b feature/gesture-classifier
git checkout -b fix/camera-latency
git checkout -b ivan/stability-filter
```

### Save your changes
```bash
# 1. See what files you changed
git status

# 2. Stage the files you want to commit
git add gesture_game/processor/gesture_classifier.py   # stage one file
git add gesture_game/                                   # stage a whole folder
git add .                                               # stage everything

# 3. Commit with a clear message
git commit -m "improve gesture classifier accuracy"

# 4. Push your branch to GitHub
git push -u origin feature/your-feature-name
```

### Keep your branch up to date with main
```bash
git checkout main
git pull origin main
git checkout feature/your-feature-name
git merge main
```

### Merge into main (via Pull Request — do NOT push directly to main)
1. Push your branch: `git push -u origin feature/your-feature-name`
2. Go to the repo on GitHub → click **"Compare & pull request"**
3. Have a teammate review it, then merge

### Useful commands
```bash
git log --oneline        # see commit history
git diff                 # see unstaged changes
git branch               # list all local branches
git checkout main        # switch back to main
```

---

## Project Structure

```
gesture_game/
├── data/
│   ├── webcam_loader.py       # Camera loader (ffmpeg-first, OpenCV fallback)
│   └── dataset_loader.py      # Optional: load HaGRID dataset for training
├── processor/
│   ├── hand_detector.py       # MediaPipe wrapper, runs per half-frame
│   ├── gesture_classifier.py  # Singleton — landmark heuristics → Attack/Defend/Heal
│   ├── stability_filter.py    # Sliding-window vote to reduce prediction flicker
│   └── game_engine.py         # Singleton — HP, rules, round state
├── ui/
│   └── renderer.py            # All cv2 drawing (HUD, HP bars, overlays)
├── main.py                    # State machine, wires all three tiers
└── ../logs/                   # Runtime logs/artifacts directory
```
