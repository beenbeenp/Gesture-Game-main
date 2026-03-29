"""
Initial ML baseline pipeline for prototype submission.

Trains a simple classifier on flattened RGB images loaded by DatasetLoader.
Expected dataset layout:
  data_root/fist, data_root/palm, data_root/peace
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gesture_game.data.dataset_loader import DatasetLoader  # noqa: E402


def main() -> None:
    data_root = os.environ.get("GESTURE_DATA_ROOT", "").strip()
    if not data_root:
        raise SystemExit("Set GESTURE_DATA_ROOT=/path/to/data_root before running.")

    X, y_str = DatasetLoader(data_root, img_size=(64, 64)).load_flat()
    if len(X) == 0:
        raise SystemExit(
            "No images loaded. Check GESTURE_DATA_ROOT and folder layout "
            "(fist/palm/peace)."
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    print("Labels:", list(encoder.classes_))
    print(classification_report(yte, pred, target_names=encoder.classes_))
    print("Confusion matrix:\n", confusion_matrix(yte, pred))


if __name__ == "__main__":
    main()
