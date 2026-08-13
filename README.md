# Hammyhamster

Point your webcam at yourself and pull faces / gestures - a hamster meme reacts live next to your camera feed.

## Setup (macOS)

```bash
git clone https://github.com/catherpiee/hammyhamster.git
cd hammyhamster
./setup.sh
```

`setup.sh` creates a virtual environment, installs dependencies, downloads the three MediaPipe models it needs (hand, face, and pose landmarkers), and launches the app. On the first run, macOS will prompt you to grant your Terminal camera access - allow it, then run `./setup.sh` again if the window doesn't open.

To run it again later without reinstalling:

```bash
source .venv/bin/activate
python3 main.py
```

Press `q` in the window to quit.

## Gestures

| Do this | You get |
|---|---|
| Nothing / no match | poker face hamster |
| Thumbs up (away from your face) | thumbs up hamster |
| Thumbs down (away from your face) | thumbs down hamster |
| Closed fist held beside your head | lollipop hamster |
| Pinch (thumb + index touching) near your face | glasses hamster |
| Index finger near your mouth | finger-near-mouth hamster |
| Index finger up, away from your mouth | nerd hamster |
| Bent elbow, wrist raised above shoulder, elbow out to the side | bicep hamster |
| Both wrists tucked together at chest height (crossed arms - hands can be hidden) | crossed-arms hamster |
| Two hands visible, no other match | truck hamster |
| Turn your head to the side | side-eye hamster |

Priority order when multiple things could apply: pinch, then fist-beside-head/thumbs, then pointer (mouth/nerd), then two-hands, then crossed-arms/bicep (pose-based fallback), then head-turn, then default.

A debug readout in the top-left corner of the camera view shows the raw gesture, head yaw, finger states, pinch/thumb/mouth-distance numbers, and crossed-arms/bicep pose signals live - useful for retuning thresholds in `main.py` if a gesture feels too sensitive or insensitive for your lighting/setup.

## Project layout

- `main.py` - the app
- `images/` - the meme images shown for each gesture
- `*.task` - MediaPipe models, downloaded by `setup.sh` (not committed)

## Requirements

- Python 3.10+
- A webcam
- macOS (camera permission prompt is macOS-specific; should also run on Linux/Windows with a webcam, just without that prompt)
