# Photo Face-Sorter

A local Python CLI tool that sorts a folder of photos into per-person folders using face recognition. It learns over repeated runs — label a new person once, and every future scan recognises them automatically with no extra effort.

All processing happens on your machine. No photos are uploaded anywhere.

---

## How it works

Each scan has two passes:

1. **Recognize** — every detected face is matched against a local database of known people. Matches are placed directly into `People/<name>/`.
2. **Discover** — unrecognized faces are clustered into provisional groups and placed in `_Review/`. You label each group once by renaming the folder. On the next scan those people are recognized automatically.

After a few runs, almost all photos sort themselves.

---

## Requirements

- macOS (Apple Silicon recommended — tested on M4 Max)
- Python 3.11+
- Internet access on first run only (to download the `buffalo_l` face recognition model)

---

## Installation

```bash
cd photo-sorter
python3 -m venv .venv
.venv/bin/pip install onnxruntime insightface pillow-heif numpy scikit-learn pyyaml opencv-python-headless
```

Then edit `config.yaml` and set `input_dir` to your photos folder:

```yaml
input_dir: "~/Pictures/Incoming"
```

---

## Workflow

All commands are run from inside the `photo-sorter/` directory.

### 1. Dry run — preview before writing anything

```bash
.venv/bin/python -m sorter.cli scan ~/Pictures/Incoming
```

Prints a summary: how many photos match known people, how many new groups were found, how many have no face. Nothing is written to disk.

### 2. Apply — detect and place files

```bash
.venv/bin/python -m sorter.cli scan ~/Pictures/Incoming --apply
```

Each photo is routed to one of:

| Destination | Condition |
|---|---|
| `People/<name>/` | Face matched a known person |
| `_Review/group_XXXX/` | Unknown face — awaiting your label |
| `_Unsorted/` | No face detected, or unreadable image |
| `_Other/` | Unsupported file type (video, doc, etc.) |

Each `_Review/group_XXXX/` folder contains the photos plus a `_faces/` subfolder of cropped face thumbnails to help you identify who it is.

### 3. Label new groups in Finder

Open `_Review/` in Finder and work through the `group_XXXX/` folders:

**Rename to a person's name:**
```
_Review/group_0001/  →  _Review/Alice/
```

**Two groups are the same person** — move the photos (not `_faces/`) from one folder into the other, delete the empty folder, rename the remaining one:
```
Move photos: group_0002/ → group_0001/
Delete: group_0002/
Rename: group_0001/ → Alice/
```

**Send to `_Unsorted`** — rename to any name starting with `_unsorted`:
```
_Review/group_0003/  →  _Review/_unsorted/
_Review/group_0004/  →  _Review/_unsorted 2/
```

**Many small groups not worth labeling** — use `prune`:
```bash
.venv/bin/python -m sorter.cli prune            # move groups with ≤ 2 photos
.venv/bin/python -m sorter.cli prune --max-size 5
```

Unlabeled `group_XXXX/` folders are skipped and left for next time.

### 4. Commit

```bash
.venv/bin/python -m sorter.cli commit
```

For each renamed group:
- Creates or merges the person in the database
- Enrolls their face embeddings for future recognition
- Moves photos into `People/<name>/`
- Cleans up the `_Review/<name>/` folder

### 5. Repeat

On the next scan, photos of already-labeled people route straight to `People/<name>/` — no review needed.

---

## All commands

| Command | Description |
|---|---|
| `scan [DIR] [--apply] [--force] [--recursive]` | Detect and route photos |
| `commit` | Enroll labeled review groups |
| `prune [--max-size N]` | Move small unlabeled groups to `_Unsorted/` |
| `list-people` | List all enrolled people with embedding counts |
| `delete-person <name> [--keep-files]` | Remove a person and re-queue their photos |
| `status` | Show database and queue stats |

---

## Supported formats

JPEG, PNG, iPhone HEIC. EXIF orientation is applied automatically.

---

## Configuration

Key settings in `config.yaml`:

```yaml
input_dir: "~/Pictures/Incoming"
placement: clone          # clone (APFS CoW, ~free) | copy | move
match_threshold: 0.50     # raise to reduce false matches, lower to be more lenient
cluster_eps: 0.45         # raise to merge more faces per group, lower to split more
max_embeddings_per_person: 20
```

The default `clone` placement uses APFS copy-on-write — placing a photo costs nearly zero extra disk space when source and `People/` are on the same volume.

---

## Privacy

Everything runs locally. No photos, embeddings, or metadata leave your machine. The face recognition model (`buffalo_l` from [InsightFace](https://github.com/deepinsight/insightface)) is downloaded once and runs fully offline.
