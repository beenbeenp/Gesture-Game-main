# Gesture Game Blog Draft

## Background: Why this problem matters

Most webcam games still rely on controllers or keyboard input. We are exploring a low-cost, controller-free interaction style where two players can compete using only hand gestures from a single webcam. This can improve accessibility and reduce hardware friction for casual play.

## Prior work and technical context

1. MediaPipe Hand Landmarker (official docs)
   - Real-time hand landmark detection for images/video streams + handedness (left/right).
   - https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
2. MediaPipe Hands (21 landmarks)
   - Infers 21 3D hand landmarks from a single frame; practical for real-time tracking.
   - https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
3. HaGRID dataset (public hand gesture dataset)
   - Dataset page: https://www.kaggle.com/datasets/kapitanov/hagrid
   - Paper: https://arxiv.org/abs/2206.08219

## What we are building

We are building a two-player lock-step PvP gesture game:
- `Attack` (fist)
- `Defend` (open hand)
- `Heal` (peace sign)

Current system status:
- Real-time webcam pipeline is implemented.
- Gesture prediction is currently heuristic landmark logic.
- Stability voting is used to reduce flicker before round resolution.

## Next steps

1. Train an initial baseline classifier on dataset images (`ml/train_baseline_sklearn.py`).
2. Compare baseline model outputs against heuristic predictions.
3. Add a short experimental log (dataset split, metrics, failure examples).
