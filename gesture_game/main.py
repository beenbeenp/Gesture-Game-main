"""
Gesture Game – entry point
Wires the three tiers together and runs the game loop.

State machine (per round):
    countdown  →  decision  →  result  →  countdown  …  →  game_over

  countdown : 3-second visual countdown displayed on screen
  decision  : 2-second "SHOW!" window — players present their gestures;
              StabilityFilter accumulates predictions
  result    : gestures locked, GameEngine resolves, result shown for 2 s
  game_over : winner screen; R restarts, Q quits
"""

import time
import cv2

from data.webcam_loader       import WebcamLoader
from processor.hand_detector  import HandDetector
from processor.gesture_classifier import GestureClassifier
from processor.stability_filter   import StabilityFilter
from processor.game_engine        import GameEngine
from ui.renderer import Renderer

# ── Timing constants (seconds) ────────────────────────────────────────
COUNTDOWN_SECS = 3
DECISION_SECS  = 2
RESULT_SECS    = 2


def split_frame(frame):
    """Return (left_half, right_half) — player 1 left, player 2 right."""
    mid = frame.shape[1] // 2
    return frame[:, :mid], frame[:, mid:]


def _pick_gesture(stability: StabilityFilter, player: int,
                  classifier: GestureClassifier, landmarks: list) -> str:
    """
    Best available gesture for one player.
    Priority: stable window vote > single-frame prediction > 'Unknown'.
    """
    stable = stability.get_stable(player)
    if stable:
        return stable
    raw = classifier.predict(landmarks[0] if landmarks else None)
    return raw


def main():
    # ── Instantiate all components ─────────────────────────────────────
    cam        = WebcamLoader()
    detector   = HandDetector(max_hands=1, detection_confidence=0.65)
    classifier = GestureClassifier()          # Singleton
    stability  = StabilityFilter(window_size=10, threshold=0.6)
    engine     = GameEngine()                 # Singleton
    renderer   = Renderer()

    # ── Initial state ──────────────────────────────────────────────────
    phase        = "countdown"
    phase_start  = time.time()

    p1_gesture   = None    # live/display gesture
    p2_gesture   = None
    p1_locked    = None    # gesture locked at end of decision window
    p2_locked    = None

    print("Gesture Game running — Q to quit, R to restart after game over.")

    while True:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)          # mirror for natural feel
        left, right = split_frame(frame)

        # ── Tier 1 → Tier 2: detect & classify ────────────────────────
        lm1 = detector.detect(left)
        lm2 = detector.detect(right)

        raw1 = classifier.predict(lm1[0] if lm1 else None)
        raw2 = classifier.predict(lm2[0] if lm2 else None)

        stability.update(1, raw1)
        stability.update(2, raw2)

        # Draw landmarks on the half-frames (debug aid)
        detector.draw_landmarks(left,  lm1)
        detector.draw_landmarks(right, lm2)

        # ── State machine ──────────────────────────────────────────────
        now     = time.time()
        elapsed = now - phase_start

        if phase == "countdown":
            countdown_val = max(1, COUNTDOWN_SECS - int(elapsed))
            p1_gesture    = _pick_gesture(stability, 1, classifier, lm1)
            p2_gesture    = _pick_gesture(stability, 2, classifier, lm2)

            if elapsed >= COUNTDOWN_SECS:
                phase       = "decision"
                phase_start = now
                stability.reset()       # fresh window for the decision

        elif phase == "decision":
            countdown_val = None
            p1_gesture    = _pick_gesture(stability, 1, classifier, lm1)
            p2_gesture    = _pick_gesture(stability, 2, classifier, lm2)

            if elapsed >= DECISION_SECS:
                # Lock gestures — prefer stable vote, fall back to best guess
                p1_locked = (stability.get_stable(1)
                             or stability.best_guess(1)
                             or "Unknown")
                p2_locked = (stability.get_stable(2)
                             or stability.best_guess(2)
                             or "Unknown")

                result = engine.resolve(p1_locked, p2_locked)
                print(f"[Round {engine.round}] {p1_locked} vs {p2_locked} → {result}")

                phase       = "result"
                phase_start = now

        elif phase == "result":
            countdown_val = None
            p1_gesture    = p1_locked
            p2_gesture    = p2_locked

            if elapsed >= RESULT_SECS:
                if engine.is_game_over():
                    phase = "game_over"
                else:
                    phase       = "countdown"
                    phase_start = now
                    stability.reset()

        elif phase == "game_over":
            countdown_val = None
            p1_gesture    = p1_locked
            p2_gesture    = p2_locked

        # ── Tier 3: render ─────────────────────────────────────────────
        game_state = {
            "hp":          engine.hp,
            "round":       engine.round,
            "last_result": engine.last_result,
        }

        if phase == "game_over":
            frame = renderer.draw_game_over(frame, engine.winner)
        else:
            frame = renderer.draw(
                frame, game_state,
                p1_gesture, p2_gesture,
                countdown_val=countdown_val if phase == "countdown" else None,
                phase=phase,
            )

        cv2.imshow("Gesture Game", frame)

        # ── Key handling ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r") and phase == "game_over":
            engine.reset()
            stability.reset()
            p1_locked = p2_locked = None
            phase       = "countdown"
            phase_start = time.time()

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
