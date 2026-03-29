# Dataset (Prototype)

We use a public hand-gesture dataset plus optional in-domain webcam samples.

## Public dataset (downloaded)

- HaGRID (HAnd Gesture Recognition Image Dataset)
- For prototype experiments, use a smaller subset (for example, HaGRID Sample 30k 384p) to keep setup practical.

## Folder layout expected by `DatasetLoader`

Create a dataset root with this structure:

```text
data_root/
  fist/   -> Attack
  palm/   -> Defend
  peace/  -> Heal
```

This matches `gesture_game/data/dataset_loader.py`:
- `fist` maps to `Attack`
- `palm` maps to `Defend`
- `peace` maps to `Heal`

## Where to put data (do not commit to git)

Keep raw image data outside this repo and point to it with:

```bash
export GESTURE_DATA_ROOT=/path/to/data_root
```

## Notes

- The full HaGRID release is very large; a subset is enough for initial prototype checkpoints.
- You can optionally combine public data with your own webcam captures for in-domain tuning.
