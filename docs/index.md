# Gesture Game Blog Draft

## Background: Why this problem matters

Most webcam games still rely on controllers or keyboard input. We are exploring a low-cost, controller-free interaction style where two players can compete using only hand gestures from a single webcam. This can improve accessibility and reduce hardware friction for casual play.

## Prior work and technical context

1. MediaPipe Hand Landmarker
   - Provides hand landmark detection for image/video streams and is practical for real-time interaction.
2. MediaPipe Hands (21 landmarks)
   - The 21-point hand representation is commonly used for gesture understanding.
3. HaGRID dataset
   - A large public hand-gesture dataset useful for initial supervised learning experiments.

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
