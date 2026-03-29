# Initial ML Pipeline (Prototype)

This folder contains the first baseline training/evaluation script for prototype grading.

## 1) Set dataset path

```bash
export GESTURE_DATA_ROOT=/path/to/data_root
```

Expected class folders inside `data_root`:
- `fist/`
- `palm/`
- `peace/`

## 2) Run baseline

From repository root:

```bash
python ml/train_baseline_sklearn.py
```

## Output

The script prints:
- class label order
- classification report (precision/recall/F1)
- confusion matrix
