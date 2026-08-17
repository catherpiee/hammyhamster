#!/usr/bin/env bash
# Sets up the hamster gesture-cam app and runs it.
set -euo pipefail
cd "$(dirname "$0")"

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "Downloading MediaPipe models..."
[ -f hand_landmarker.task ] || curl -sL -o hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
[ -f face_landmarker.task ] || curl -sL -o face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
[ -f pose_landmarker.task ] || curl -sL -o pose_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

echo "Starting app..."
python3 main.py
