# Photo Face-Sorter — User Guide

A local tool that sorts photos into per-person folders using face recognition. It learns over time: label a person once, and every future run recognises them automatically.

---

## Quick Start

```bash
# 1. Dry run — preview only, nothing written to disk
.venv/bin/python -m sorter.cli scan ~/Pictures/Incoming

# 2. Apply — detect faces and place files
.venv/bin/python -m sorter.cli scan ~/Pictures/Incoming --apply
```

Open `_Review/` in Finder. Each `group_XXXX/` folder is a cluster of unknown faces. Look at `_faces/` inside each group for cropped thumbnails to identify the person. Then:

**Label a group** — rename the folder to the person's name:
```
_Review/group_0001/  →  _Review/Alice/
```

**Merge two groups (same person)** — move the photos (not `_faces/`) from one group into the other, delete the empty folder, then rename the remaining one:
```
Move photos from group_0002/ into group_0001/
Delete group_0002/
Rename group_0001/ → Alice/
```

**Send a group to `_Unsorted`** — rename it to any name starting with `_unsorted`:
```
_Review/group_0003/  →  _Review/_unsorted/
_Review/group_0004/  →  _Review/_unsorted 2/
```

**Move many small groups to `_Unsorted` at once** — use `prune` before committing:
```bash
# Move all unlabeled groups with ≤ 2 photos (default)
.venv/bin/python -m sorter.cli prune

# Or set your own threshold
.venv/bin/python -m sorter.cli prune --max-size 5
```

Once done labeling, commit:
```bash
.venv/bin/python -m sorter.cli commit
```

---

## Setup

All commands are run from inside the `photo-sorter/` directory.

**1. Install dependencies (first time only)**
```bash
.venv/bin/pip install onnxruntime insightface pillow-heif numpy scikit-learn pyyaml opencv-python-headless
```

The face-recognition model (`buffalo_l`) downloads automatically on the first `scan` run — requires internet access that one time only.

**2. Edit `config.yaml`**

Set `input_dir` to the folder where your new photos live:
```yaml
input_dir: "~/Pictures/Incoming"
```

All other defaults are sensible to start. See [Configuration](#configuration) for tuning options.

---

## Basic Workflow

### Step 1 — Dry run (preview only)
```bash
# Uses input_dir from config.yaml
.venv/bin/python -m sorter.cli scan

# Or specify a folder directly on the command line
.venv/bin/python -m sorter.cli scan ~/Pictures/Incoming
```
Prints a summary of what would happen: how many photos match known people, how many new groups were found, how many have no face. **Nothing is written to disk.**

### Step 2 — Apply
```bash
# Uses input_dir from config.yaml
.venv/bin/python -m sorter.cli scan --apply

# Or specify a folder directly
.venv/bin/python -m sorter.cli scan ~/Pictures/Incoming --apply
```
Processes photos and places them:
- **Known person** → `People/<name>/`
- **Unknown face** → `_Review/group_XXXX/` (with face crops in `_faces/` subfolder)
- **No face detected** → `_Unsorted/`

### Step 3 — Label new groups in Finder

Open `_Review/` in Finder. Each `group_XXXX/` folder is a cluster of photos that appear to show the same unknown person. Look at the `_faces/` subfolder inside each group for cropped face thumbnails to help you identify them.

Rename each folder to the person's name:
```
_Review/group_0001/  →  _Review/Alice/
_Review/group_0002/  →  _Review/Bob/
```

**If two groups are the same person:** move the photos (`.jpg`, `.heic`, etc.) from one folder into the other in Finder — do not move the `_faces/` subfolder, it's just preview thumbnails. Delete the now-empty source folder, then rename the remaining one to the person's name.

**If a group should go to `_Unsorted` instead** (blurry faces, strangers, irrelevant): rename it to any name starting with `_unsorted` — e.g. `_unsorted`, `_unsorted 2`, `_unsorted_strangers`. You can have as many as you need since they just need to start with `_unsorted`.

**If you want to discard a group entirely** (don't need the photos at all): delete the folder. Originals are never touched — only clones live in `_Review/`.

**If many small groups have only a handful of photos and aren't worth labeling:** use the `prune` command instead of renaming them one by one (see below).

Leave any `group_XXXX/` folders you haven't decided on yet — they'll be skipped on commit.

### Step 4 — Commit
```bash
.venv/bin/python -m sorter.cli commit
```
For each renamed group:
- Creates or merges the person in the database
- Enrolls their face embeddings (so they're recognised on future scans)
- Moves photos into `People/<name>/` and cleans up the `_Review/<name>/` folder

Unlabeled `group_XXXX/` folders are left untouched for next time.

### Step 5 — Repeat

On the next run, photos of people you've already labeled go straight to `People/<name>/` with no review step needed.

---

## Commands

### `scan [INPUT_DIR] [--apply]`

| Argument | Description |
|---|---|
| `INPUT_DIR` | Override the input directory from config (optional) |
| `--apply` | Actually place files; without it, only a summary is printed |

```bash
# Dry run on config input_dir
.venv/bin/python -m sorter.cli scan

# Dry run on a specific folder
.venv/bin/python -m sorter.cli scan ~/Downloads/Photos

# Apply
.venv/bin/python -m sorter.cli scan --apply
.venv/bin/python -m sorter.cli scan ~/Downloads/Photos --apply
```

Runs are **idempotent** — files already processed (tracked by content hash) are skipped even if renamed or duplicated.

### `prune [--max-size N]`

```bash
# Move all unlabeled groups with ≤ 2 photos to _Unsorted/ (default)
.venv/bin/python -m sorter.cli prune

# Move all unlabeled groups with ≤ 5 photos
.venv/bin/python -m sorter.cli prune --max-size 5
```

Scans `_Review/` for unlabeled `group_XXXX/` folders and moves any with ≤ N photos directly to `_Unsorted/`, discarding their pending embeddings. Useful when many singleton or small groups aren't worth labeling. Only affects unlabeled groups — any folder you've already renamed is left alone.

### `commit`

```bash
.venv/bin/python -m sorter.cli commit
```

Always moves files out of `_Review/` into their destination (`People/<name>/` or `_Unsorted/`). Photos that cannot be processed are left in `_Review/` untouched.

### `delete-person <name> [--keep-files]`

```bash
# Remove person from DB and delete their People/<name>/ folder
.venv/bin/python -m sorter.cli delete-person Alice

# Remove from DB only, keep the folder
.venv/bin/python -m sorter.cli delete-person Alice --keep-files
```

Removes a person completely: deletes their embeddings and person record from the database, and re-queues their photos so the next `scan --apply` picks them up again as unknowns. By default also deletes `People/<name>/`; use `--keep-files` to leave it.

Requires typing the exact person name to confirm before anything is deleted.

### `status`

```bash
.venv/bin/python -m sorter.cli status
```

Prints:
- Number of known people and stored embeddings
- Number of processed files and last run time
- Groups currently in `_Review/` (labeled vs. unlabeled, with photo counts)
- Count of photos in `_Unsorted/`

---

## Output Layout

```
photo-sorter/
  People/
    Alice/          ← all photos routed to Alice
    Bob/
    …
  _Review/
    group_0003/     ← unlabeled, awaiting your decision
      photo1.jpg
      photo2.jpg
      _faces/       ← cropped face thumbnails to help identification
        photo1_face.jpg
        photo2_face.jpg
      .photosort-group.json
  _Unsorted/        ← photos with no detectable face, or groups you sent here
  _Other/           ← non-image files (videos, docs, etc.) copied here during scan
  faces.sqlite      ← database (people, embeddings, history)
  run_log.jsonl     ← timestamped log of every apply/commit run
```

---

## File Placement Modes

Set `placement` in `config.yaml`:

| Mode | Behaviour | Disk cost |
|---|---|---|
| `clone` (default) | APFS copy-on-write clone — near-zero extra space if source and `People/` are on the same volume | ~0 |
| `copy` | Standard file copy | Full copy |
| `move` | Moves originals out of the input folder | ~0 |

> **Note:** `clone` falls back to `copy` automatically if source and destination are on different volumes, or if the filesystem is not APFS.

---

## Configuration Reference

`config.yaml` — all paths support `~` expansion.

```yaml
input_dir: "~/Pictures/Incoming"   # where new photos live
output_dir: "./People"             # sorted output
review_dir: "./_Review"            # unknown groups awaiting labels
unsorted_dir: "./_Unsorted"        # no-face photos
other_dir: "./_Other"              # unsupported file types (videos, docs, etc.)
db_path: "./faces.sqlite"

device: cpu                        # cpu | coreml | cuda

det_size: [640, 640]               # detection input resolution
min_det_score: 0.5                 # ignore detections below this confidence

match_threshold: 0.50              # cosine similarity to count as a known person
cluster_eps: 0.45                  # DBSCAN distance threshold for grouping unknowns
cluster_min_samples: 2             # minimum photos to form a cluster (singletons still kept)

placement: clone                   # clone | copy | move
enroll_recognized: false           # add high-confidence matches back to the person's embeddings
max_embeddings_per_person: 20      # cap on stored face vectors per person
```

### Tuning thresholds

If **different people are being merged** into the same folder:
- Raise `match_threshold` (e.g. `0.55`) — requires closer match to call someone known
- Lower `cluster_eps` (e.g. `0.40`) — requires faces to be more similar to cluster together

If **the same person is appearing in too many separate groups**:
- Lower `match_threshold` (e.g. `0.45`)
- Raise `cluster_eps` (e.g. `0.50`)

---

## Supported Formats

JPEG (`.jpg`, `.jpeg`), PNG (`.png`), iPhone HEIC (`.heic`, `.heif`).

EXIF orientation is applied automatically, so sideways phone photos are handled correctly.

---

## Tips

- **Two groups are the same person:** in Finder, move the photos (`.jpg`, `.heic`, etc.) from one group folder into the other — ignore the `_faces/` subfolders, they are just preview thumbnails and `commit` skips them. Delete the now-empty source group folder, rename the remaining one to the person's name, and run `commit`.
- **Multiple groups to send to `_Unsorted`:** rename each one with a unique name starting with `_unsorted` — e.g. `_unsorted`, `_unsorted 2`, `_unsorted_strangers`. All will be routed to `_Unsorted/` on commit.
- **A review group is someone already committed:** rename the group to the existing person's name. `commit` merges their embeddings into the existing person record, improving future recognition.
- **Multiple people in a photo:** the photo is routed to the folder of the largest (most prominent) face. The other faces are still recorded for recognition — those people become recognisable in other photos.
- **Originals are never moved** unless you set `placement: move`. Everything in `People/` and `_Review/` is a clone or copy.
- **Run `status` after `scan --apply`** to see a quick summary of what's waiting in review before opening Finder.
- **Incorrect person / want to start over:** use `delete-person <name>` to wipe their embeddings and re-queue their photos. On the next scan they'll appear in `_Review/` as unknowns again.
