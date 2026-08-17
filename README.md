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

Press `q` or `Esc` in the window to quit (or just close the window, or hit Ctrl+C in the terminal) - it cleans up the camera and models before exiting either way. Press `d` to toggle the debug readout on/off.

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
| One hand on each cheek | shy hamster |
| Hands clasped together at mouth/chin height | thinking hamster |
| Hands clasped together at chest height, below your face | hug hamster |
| Head tilted down | sad hamster |
| Two hands visible, no other match | truck hamster |
| Turn your head to the side | side-eye hamster |

Priority order when multiple things could apply: pinch, then fist-beside-head/thumbs, then pointer (mouth/nerd), then shy/thinking/hug (two-hand shape+position), then crossed-arms/bicep (pose-based fallback), then two-hands, then head-tilt-down (sad), then head-turn (side-eye), then default.

The "sad" gesture reads head pitch off the same face-transformation-matrix trick `side_eye` uses for yaw; unlike yaw, the sign wasn't verified against a live camera during development, so if it triggers on an upward tilt instead of downward, flip the sign in `head_pitch_degrees` in `main.py`.

A debug readout in the top-left corner of the camera view shows head yaw/pitch, finger states, pinch/thumb/mouth-distance numbers, crossed-arms/bicep pose signals, and (when two hands are visible) the shy/thinking/hug distance numbers live - useful for retuning thresholds in `main.py` if a gesture feels too sensitive or insensitive for your lighting/setup. It's on by default; press `d` to hide it for a cleaner view. The current gesture is always shown as a pill in the window's header, regardless of the debug toggle.

## Project layout

- `main.py` - the app
- `images/` - the meme images shown for each gesture
- `*.task` - MediaPipe models, downloaded by `setup.sh` (not committed)

## Requirements

- Python 3.10+
- A webcam
- macOS (camera permission prompt is macOS-specific; should also run on Linux/Windows with a webcam, just without that prompt)
