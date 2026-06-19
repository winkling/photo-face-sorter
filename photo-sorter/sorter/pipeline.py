import json
import os
import re
import shutil
from datetime import datetime, timezone

import cv2
import numpy as np

from .config import load_config
from .db import (
    open_db, now_iso,
    load_known_embeddings, file_already_processed, record_processed_file,
    get_person_by_name, delete_person, delete_processed_files_for_person,
    get_or_create_person, add_embedding, count_embeddings_for_person,
    get_lowest_det_score_embedding, delete_embedding,
    insert_pending_embedding, get_pending_embeddings,
    get_pending_embeddings_by_filenames,
    delete_pending_embeddings, delete_pending_embeddings_by_ids,
)
from .images import load_bgr, iter_images, iter_other_files, crop_face
from .detect import make_app, detect_faces, prominent_face
from .match import best_match
from .cluster import cluster, assign_groups
from .placement import place_file
from .utils import sha256_file


def _log_run(cfg: dict, entry: dict):
    log_path = os.path.join(os.path.dirname(cfg["db_path"]), "run_log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps({"ts": now_iso(), **entry}) + "\n")


def _enroll_embedding(conn, person_id: int, vec: np.ndarray, source_path: str,
                      det_score: float, max_per_person: int):
    count = count_embeddings_for_person(conn, person_id)
    if count < max_per_person:
        add_embedding(conn, person_id, vec, source_path, det_score)
    else:
        lowest = get_lowest_det_score_embedding(conn, person_id)
        if lowest and lowest[1] < det_score:
            delete_embedding(conn, lowest[0])
            add_embedding(conn, person_id, vec, source_path, det_score)


def run_scan(cfg: dict, input_dir: str, apply: bool, force: bool = False, recursive: bool = False):
    conn = open_db(cfg["db_path"])
    mat, person_ids, person_names = load_known_embeddings(conn)

    app = make_app(cfg["device"], cfg["det_size"])

    no_face = []
    load_error = []  # (path, sha) — supported extension but failed to open
    known = []       # (path, person_id, person_name, emb, det_score, sim)
    unknown = []     # (path, prominent_emb, prominent_det_score, all_faces)

    other_files = list(iter_other_files(input_dir, recursive=recursive))
    images = list(iter_images(input_dir, recursive=recursive))
    total = len(images)
    skipped = 0

    for i, path in enumerate(images, 1):
        print(f"\r  [{i}/{total}] {os.path.basename(path)[:50]}\033[K", end="", flush=True)
        sha = sha256_file(path)
        if not force and file_already_processed(conn, sha):
            skipped += 1
            continue
        try:
            bgr = load_bgr(path)
        except Exception as e:
            print(f"\n  WARN: could not load {os.path.basename(path)}: {e}")
            load_error.append((path, sha))
            continue
        faces = detect_faces(app, bgr, cfg["min_det_score"])
        if not faces:
            no_face.append((path, sha))
            continue
        prom = prominent_face(faces)
        pid, sim = best_match(prom.normed_embedding, mat, person_ids, cfg["match_threshold"])
        if pid is not None:
            known.append((path, sha, pid, person_names[pid], prom.normed_embedding, prom.det_score, sim))
        else:
            unknown.append((path, sha, prom.normed_embedding, prom.det_score, faces))

    print()  # newline after progress

    # cluster unknowns
    group_map = {}  # index -> group_id
    if unknown:
        embs = np.stack([u[2] for u in unknown])
        labels = cluster(embs, cfg["cluster_eps"], cfg["cluster_min_samples"])
        group_map = assign_groups(labels)

    # --- summary ---
    group_counts = {}
    for i, gid in group_map.items():
        group_counts[gid] = group_counts.get(gid, 0) + 1

    print(f"\n=== Scan summary ({'DRY RUN' if not apply else 'APPLYING'}) ===")
    print(f"  Images found:   {total}")
    print(f"  Skipped:        {skipped} (already processed)")
    print(f"  Auto-sorted:    {len(known)} (known people)")
    print(f"  New groups:     {len(group_counts)} groups from {len(unknown)} unknown photos")
    for gid, cnt in sorted(group_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {gid}: {cnt} photo(s)")
    print(f"  No face:        {len(no_face)}")
    print(f"  Load errors:    {len(load_error)} (corrupt/unreadable, will go to _Unsorted/)")
    print(f"  Other files:    {len(other_files)} (unsupported formats, will be copied to _Other/)")

    if not apply:
        conn.close()
        return

    # --- apply ---
    placement_mode = cfg["placement"]
    output_dir = cfg["output_dir"]
    review_dir = cfg["review_dir"]
    unsorted_dir = cfg["unsorted_dir"]
    other_dir = cfg["other_dir"]
    max_emb = cfg["max_embeddings_per_person"]

    counts = {"known": 0, "review": 0, "no_face": 0, "other": 0}

    # unsupported files → _Other/
    for path in other_files:
        dst = os.path.join(other_dir, os.path.basename(path))
        place_file(path, dst, placement_mode)
        counts["other"] += 1

    # known → People/<name>/
    for (path, sha, pid, pname, emb, det_score, sim) in known:
        dst = os.path.join(output_dir, pname, os.path.basename(path))
        place_file(path, dst, placement_mode)
        record_processed_file(conn, sha, path, f"person:{pname}")
        if cfg.get("enroll_recognized"):
            _enroll_embedding(conn, pid, emb, path, det_score, max_emb)
        counts["known"] += 1

    # no face → _Unsorted/
    for (path, sha) in no_face:
        dst = os.path.join(unsorted_dir, os.path.basename(path))
        place_file(path, dst, placement_mode)
        record_processed_file(conn, sha, path, "no_face")
        counts["no_face"] += 1

    # load errors → _Unsorted/
    for (path, sha) in load_error:
        dst = os.path.join(unsorted_dir, os.path.basename(path))
        place_file(path, dst, placement_mode)
        record_processed_file(conn, sha, path, "no_face")
        counts["no_face"] += 1

    # unknown → _Review/<group_id>/
    group_paths = {}  # group_id -> [source_paths]
    for i, (path, sha, emb, det_score, all_faces) in enumerate(unknown):
        gid = group_map[i]
        group_paths.setdefault(gid, []).append(path)

        # place photo
        dst = os.path.join(review_dir, gid, os.path.basename(path))
        place_file(path, dst, placement_mode)

        # save face crop
        try:
            bgr = load_bgr(path)
            prom = prominent_face(all_faces)
            crop = crop_face(bgr, prom.bbox)
            crops_dir = os.path.join(review_dir, gid, "_faces")
            os.makedirs(crops_dir, exist_ok=True)
            crop_name = os.path.splitext(os.path.basename(path))[0] + "_face.jpg"
            cv2.imwrite(os.path.join(crops_dir, crop_name), crop)
        except Exception as e:
            print(f"\n  WARN: could not save crop for {path}: {e}")

        # pending embedding
        insert_pending_embedding(conn, gid, emb, path, det_score)
        record_processed_file(conn, sha, path, f"review:{gid}")
        counts["review"] += 1

    # write group marker files
    for gid, paths in group_paths.items():
        marker = os.path.join(review_dir, gid, ".photosort-group.json")
        with open(marker, "w") as f:
            json.dump({"group_id": gid, "source_files": paths}, f, indent=2)

    conn.commit()
    conn.close()

    _log_run(cfg, {"command": "scan", "apply": True,
                   "known": counts["known"], "review": counts["review"],
                   "no_face": counts["no_face"], "other": counts["other"]})

    print(f"\n  Placed {counts['known']} known, {counts['review']} to review, "
          f"{counts['no_face']} unsorted, {counts['other']} other.")


_GROUP_RE = re.compile(r"^group_\d{4}$")


def run_commit(cfg: dict):
    conn = open_db(cfg["db_path"])
    review_dir = cfg["review_dir"]
    output_dir = cfg["output_dir"]
    max_emb = cfg["max_embeddings_per_person"]

    if not os.path.isdir(review_dir):
        print("No _Review directory found.")
        conn.close()
        return

    committed = 0
    skipped = 0

    for folder_name in sorted(os.listdir(review_dir)):
        folder_path = os.path.join(review_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if _GROUP_RE.match(folder_name):
            skipped += 1
            continue

        marker_path = os.path.join(folder_path, ".photosort-group.json")
        if not os.path.exists(marker_path):
            print(f"  WARN: {folder_name}/ has no .photosort-group.json — skipping")
            continue

        with open(marker_path) as f:
            marker = json.load(f)
        group_id = marker["group_id"]
        person_name = folder_name
        is_unsorted = folder_name.lower().startswith("_unsorted")

        placed = 0

        if is_unsorted:
            # discard pending embeddings — these photos are intentionally unsorted
            delete_pending_embeddings(conn, group_id)
            dest_dir = cfg["unsorted_dir"]
            for fname in os.listdir(folder_path):
                if fname.startswith(".") or fname == "_faces":
                    continue
                src = os.path.join(folder_path, fname)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(dest_dir, fname)
                place_file(src, dst, "move")
                placed += 1
            shutil.rmtree(folder_path)
            committed += 1
            print(f"  Routed to _Unsorted ({group_id}, {placed} photos)")
        else:
            # merge-or-create person
            pid = get_or_create_person(conn, person_name)

            # enroll pending embeddings for this group
            rows = get_pending_embeddings(conn, group_id)
            enrolled_ids = [r[0] for r in rows]
            for (_, vec_blob, src_path, det_score) in rows:
                vec = np.frombuffer(vec_blob, dtype="<f4")
                _enroll_embedding(conn, pid, vec, src_path, det_score, max_emb)
            delete_pending_embeddings(conn, group_id)

            # also pick up embeddings from photos moved in from other groups
            photo_filenames = [
                f for f in os.listdir(folder_path)
                if not f.startswith(".") and f != "_faces"
                and os.path.isfile(os.path.join(folder_path, f))
            ]
            extra_rows = [
                r for r in get_pending_embeddings_by_filenames(conn, photo_filenames)
                if r[0] not in set(enrolled_ids)
            ]
            if extra_rows:
                extra_ids = [r[0] for r in extra_rows]
                for (_, vec_blob, src_path, det_score) in extra_rows:
                    vec = np.frombuffer(vec_blob, dtype="<f4")
                    _enroll_embedding(conn, pid, vec, src_path, det_score, max_emb)
                delete_pending_embeddings_by_ids(conn, extra_ids)
                print(f"    + {len(extra_rows)} embedding(s) from merged groups")

            # place photos into People/<name>/
            for fname in os.listdir(folder_path):
                if fname.startswith(".") or fname == "_faces":
                    continue
                src = os.path.join(folder_path, fname)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(output_dir, person_name, fname)
                place_file(src, dst, "move")
                placed += 1

            # remove review folder
            shutil.rmtree(folder_path)
            committed += 1
            print(f"  Committed '{person_name}' ({group_id}, {placed} photos, {len(rows)} embeddings)")

    conn.commit()
    conn.close()

    _log_run(cfg, {"command": "commit", "committed": committed, "skipped_unlabeled": skipped})
    print(f"\nDone. {committed} group(s) committed, {skipped} still unlabeled (skipped).")


def run_delete_person(cfg: dict, name: str, keep_files: bool):
    conn = open_db(cfg["db_path"])

    row = get_person_by_name(conn, name)
    if not row:
        print(f"Person '{name}' not found in database.")
        conn.close()
        return

    person_id = row[0]
    emb_count = conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE person_id = ?", (person_id,)
    ).fetchone()[0]
    file_count = conn.execute(
        "SELECT COUNT(*) FROM processed_files WHERE result = ?", (f"person:{name}",)
    ).fetchone()[0]

    print(f"  Person:     {name}")
    print(f"  Embeddings: {emb_count}")
    print(f"  Processed files to re-queue: {file_count}")

    people_folder = os.path.join(cfg["output_dir"], name)
    if not keep_files and os.path.isdir(people_folder):
        print(f"  Folder:     {people_folder} (will be deleted)")
    elif keep_files:
        print(f"  Folder:     {people_folder} (kept)")

    confirm = input("\nType the person's name to confirm deletion: ").strip()
    if confirm != name:
        print("Cancelled — name did not match.")
        conn.close()
        return

    delete_person(conn, person_id)
    delete_processed_files_for_person(conn, name)
    conn.commit()
    conn.close()

    if not keep_files and os.path.isdir(people_folder):
        shutil.rmtree(people_folder)
        print(f"  Deleted folder: {people_folder}")

    print(f"\nDone. '{name}' removed from database. "
          f"Their {file_count} photo(s) will be re-queued on the next scan.")


def run_prune(cfg: dict, max_size: int):
    review_dir = cfg["review_dir"]
    unsorted_dir = cfg["unsorted_dir"]
    conn = open_db(cfg["db_path"])

    if not os.path.isdir(review_dir):
        print("No _Review directory found.")
        conn.close()
        return

    pruned = 0
    skipped = 0

    for folder_name in sorted(os.listdir(review_dir)):
        folder_path = os.path.join(review_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if not _GROUP_RE.match(folder_name):
            skipped += 1
            continue

        photos = [
            f for f in os.listdir(folder_path)
            if not f.startswith(".") and f != "_faces"
            and os.path.isfile(os.path.join(folder_path, f))
        ]
        if len(photos) > max_size:
            skipped += 1
            continue

        marker_path = os.path.join(folder_path, ".photosort-group.json")
        if os.path.exists(marker_path):
            with open(marker_path) as f:
                group_id = json.load(f)["group_id"]
            delete_pending_embeddings(conn, group_id)

        os.makedirs(unsorted_dir, exist_ok=True)
        for fname in photos:
            shutil.move(os.path.join(folder_path, fname), os.path.join(unsorted_dir, fname))

        shutil.rmtree(folder_path)
        pruned += 1
        print(f"  Pruned {folder_name} ({len(photos)} photo(s)) → _Unsorted/")

    conn.commit()
    conn.close()

    print(f"\nDone. {pruned} small group(s) moved to _Unsorted/, {skipped} kept.")


def run_status(cfg: dict):
    conn = open_db(cfg["db_path"])

    people_count = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
    emb_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    processed_count = conn.execute("SELECT COUNT(*) FROM processed_files").fetchone()[0]
    last_run = conn.execute("SELECT MAX(processed_at) FROM processed_files").fetchone()[0]

    conn.close()

    # _Review pending
    review_dir = cfg["review_dir"]
    review_groups = {}
    if os.path.isdir(review_dir):
        for folder_name in os.listdir(review_dir):
            folder_path = os.path.join(review_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            count = sum(
                1 for f in os.listdir(folder_path)
                if not f.startswith(".") and f != "_faces"
                and os.path.isfile(os.path.join(folder_path, f))
            )
            review_groups[folder_name] = count

    # _Unsorted
    unsorted_dir = cfg["unsorted_dir"]
    unsorted_count = 0
    if os.path.isdir(unsorted_dir):
        unsorted_count = sum(1 for f in os.listdir(unsorted_dir)
                             if os.path.isfile(os.path.join(unsorted_dir, f)))

    print("=== Photo Sorter Status ===")
    print(f"  Known people:      {people_count}")
    print(f"  Stored embeddings: {emb_count}")
    print(f"  Processed files:   {processed_count}")
    print(f"  Last run:          {last_run or 'never'}")
    print(f"  Unsorted photos:   {unsorted_count}")
    if review_groups:
        print(f"  Review pending:    {len(review_groups)} group(s)")
        for gname, cnt in sorted(review_groups.items()):
            label = "(unlabeled)" if _GROUP_RE.match(gname) else "(labeled, not committed)"
            print(f"    {gname}: {cnt} photo(s) {label}")
    else:
        print("  Review pending:    none")
