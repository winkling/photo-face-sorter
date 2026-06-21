# Photo Face-Sorter — Codebase Reference

A local Python CLI tool that sorts photos into per-person folders using face recognition. Learns over repeated runs: people are labeled once via `_Review/`, then recognised automatically on future scans.

---

## Environment

- **Platform:** macOS, Apple Silicon, APFS filesystem
- **Language:** Python 3.11+, venv at `photo-sorter/.venv/`
- **Compute:** CPU only (`device: cpu` in config). `coreml`/`cuda` are config options but untested.
- **Run from:** `photo-sorter/` directory

---

## Project Layout

```
photo-sorter/
  config.yaml          # paths, thresholds, placement mode
  faces.sqlite         # runtime — gitignored
  run_log.jsonl        # timestamped JSONL log of every apply/commit — gitignored
  sorter/
    __init__.py
    config.py          # load/validate config.yaml, expand ~ in paths
    db.py              # SQLite layer — open_db, all CRUD helpers, embedding serialization
    images.py          # load_bgr (JPEG/PNG/WebP/BMP/HEIC + EXIF), iter_images, iter_other_files, crop_face
    detect.py          # InsightFace buffalo_l wrapper — make_app, detect_faces, prominent_face
    match.py           # cosine dot-product match against known-people matrix
    cluster.py         # DBSCAN clustering + noise→singleton group assignment
    placement.py       # place_file: clonefile(2) / shutil.copy2 / shutil.move
    utils.py           # sha256_file
    pipeline.py        # run_scan, run_commit, run_status
    cli.py             # argparse entry point
  People/              # output — gitignored
  _Review/             # pending groups — gitignored
  _Unsorted/           # no-face photos — gitignored
  _Other/              # unsupported file types — gitignored
```

---

## Config (`config.yaml`)

```yaml
input_dir: "~/Pictures/Incoming"
output_dir: "./People"
review_dir: "./_Review"
unsorted_dir: "./_Unsorted"
other_dir: "./_Other"
db_path: "./faces.sqlite"

device: cpu                           # cpu | coreml | cuda
det_size: [640, 640]
min_det_score: 0.5

match_threshold: 0.50                 # cosine sim to count as known (TUNE)
cluster_eps: 0.45                     # DBSCAN cosine distance (TUNE)
cluster_min_samples: 2

placement: clone                      # clone | copy | move
enroll_recognized: false
max_embeddings_per_person: 20
```

All path keys expand `~`. Thresholds are intentionally exposed for tuning.

---

## Data Model (SQLite)

Four tables in `faces.sqlite`:

- **`people`** — `id, name, created_at`
- **`embeddings`** — `id, person_id, vec BLOB, source_path, det_score, created_at`. `vec` is 512 × float32 little-endian (`vec.astype('<f4').tobytes()`). Always stores `face.normed_embedding` so cosine = dot product.
- **`processed_files`** — `sha256 PK, path, result, processed_at`. Result: `person:<name>` | `review:<group_id>` | `no_face`. Keyed on SHA-256 of file content for idempotency even if files are renamed.
- **`pending_embeddings`** — `id, group_id, vec, source_path, det_score`. Staged between `scan` and `commit`. Deleted on commit.

---

## Commands

### `scan [INPUT_DIR] [--dry-run] [--force] [--recursive]`

```bash
.venv/bin/python -m sorter.cli scan
.venv/bin/python -m sorter.cli scan ~/Pictures/Incoming
.venv/bin/python -m sorter.cli scan --recursive   # include subfolders
.venv/bin/python -m sorter.cli scan --force       # re-process already-processed files
.venv/bin/python -m sorter.cli scan --dry-run     # preview only, nothing written
```

**With `--dry-run`** — prints summary only, nothing written.

**Without `--dry-run` (default):**
1. Copies unsupported file types (non-image) to `_Other/`.
2. For each supported image not already in `processed_files` (by SHA-256), skipped unless `--force`:
   - No faces / load error → `_Unsorted/`
   - Known face (cosine sim ≥ `match_threshold`) → `People/<name>/`
   - Unknown face → collected for DBSCAN clustering → `_Review/group_XXXX/`
3. Unknown groups get: photos placed, face crops in `_faces/`, marker `.photosort-group.json`, embeddings in `pending_embeddings`.
4. Records every file in `processed_files`.

### `commit`

```bash
.venv/bin/python -m sorter.cli commit
```

Walks `_Review/` subfolders. Always **moves** files (not clone/copy).

- **`group_XXXX/` (unrenamed)** → skip, leave for later.
- **`_unsorted*` (any name starting with `_unsorted`)** → move photos to `_Unsorted/`, discard pending embeddings, delete folder. Multiple such folders allowed (e.g. `_unsorted`, `_unsorted 2`).
- **Any other name** → treated as a person name. Merge-or-create the person, enroll pending embeddings (by `group_id`), also pick up embeddings for any photos manually moved in from other groups (matched by filename). Move photos to `People/<name>/`, delete folder.

### `list-people`

```bash
.venv/bin/python -m sorter.cli list-people
```

Lists all enrolled people (DB id, name, embedding count, enrolled date), sorted alphabetically.

### `status`

```bash
.venv/bin/python -m sorter.cli status
```

Prints: known people, stored embeddings, processed files, last run time, per-group review queue, unsorted count.

---

## Key Behaviours

**Idempotency:** `scan` skips files already in `processed_files` by SHA-256 — safe to re-run on the same folder.

**Merging two `_Review` groups (same person):** move photos (not `_faces/`) from one group folder into the other in Finder, delete the empty folder, rename the remaining one. `commit` picks up the orphaned embeddings via filename matching (`get_pending_embeddings_by_filenames`).

**Merge-or-create on commit:** if a person name already exists in the DB, their embeddings are merged — this corrects recognition misses and improves future matching.

**Embedding cap:** `max_embeddings_per_person` enforced by keeping highest `det_score` entries.

**APFS clone:** `placement.py` uses `clonefile(2)` via ctypes. Falls back to `shutil.copy2` on cross-volume or non-APFS. `commit` always uses move regardless of the `placement` config.

---

## Dependencies

```
pip install -r requirements.txt
```

`buffalo_l` model downloads on first `scan` run (requires internet once).
