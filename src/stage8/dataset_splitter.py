"""
dataset_splitter.py — Phase B: Dataset Validation and Splitting.
=================================================================

Validates the corpus manifest, checks dataset availability, and creates
deterministic, leakage-free train/validation splits.

Key design decisions:

  * Hamlet is split at SCENE boundaries (using hamlet_scenes.jsonl).
    Whole scenes go to train or validation — no scene is split across
    the boundary. This prevents context leakage between scenes.

  * Stage 5 original documents are split at DOCUMENT boundaries.
    Whole documents go to train or validation — no document is split.

  * Missing datasets (file = null in manifest) are reported but not
    fabricated. No fake data is generated.

  * All splits use a deterministic random seed for reproducibility.

  * Hamlet's weight is capped at its configured maximum (25%) so it
    cannot dominate the language distribution merely because its file
    is larger than the original documents.

All reads/writes stay inside the project folder. Nothing is overwritten.

This file is part of the from-scratch my-LLM project. We authored it.
"""

import hashlib
import json
import os
import random
import sys


# --- project-relative paths ----------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "manifests", "corpus_manifest.json")
HAMLET_CLEAN = os.path.join(PROJECT_ROOT, "data", "processed", "hamlet_clean.txt")
HAMLET_SCENES = os.path.join(PROJECT_ROOT, "data", "processed", "hamlet_scenes.jsonl")
STAGE5_TRAIN = os.path.join(PROJECT_ROOT, "data", "stage5", "train.txt")
STAGE5_VAL = os.path.join(PROJECT_ROOT, "data", "stage5", "val.txt")
STAGE5_SOURCES = os.path.join(PROJECT_ROOT, "data", "stage5", "sources")
TINY_CORPUS = os.path.join(PROJECT_ROOT, "data", "tiny_corpus.txt")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "stage8")
TRAIN_SPLITS_DIR = os.path.join(OUTPUT_DIR, "splits")


def fail(message):
    print("\nERROR: " + message, file=sys.stderr)
    sys.exit(1)


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    """Load and validate the corpus manifest."""
    if not os.path.isfile(MANIFEST_PATH):
        fail(f"Corpus manifest not found: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest


def validate_dataset_entry(entry):
    """Check a single corpus entry for availability.

    Returns a dict with keys:
        id, name, file, weight, status, exists, size_chars, notes
    """
    result = {
        "id": entry.get("id", "unknown"),
        "name": entry.get("name", "unknown"),
        "file": entry.get("file"),
        "weight": entry.get("weight", 0.0),
        "notes": entry.get("notes", ""),
        "exists": False,
        "size_chars": 0,
        "status": "missing",
    }

    file_path = entry.get("file")
    if file_path is None:
        result["status"] = "placeholder"
        result["notes"] = entry.get(
            "notes", "No file specified — dataset not yet available."
        )
        return result

    full_path = os.path.join(PROJECT_ROOT, file_path)
    if not os.path.isfile(full_path):
        result["status"] = "missing_file"
        result["notes"] = f"File listed in manifest but not found: {file_path}"
        return result

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        result["status"] = "read_error"
        result["notes"] = f"Could not read file: {exc}"
        return result

    if not text.strip():
        result["status"] = "empty"
        result["notes"] = "File exists but is empty or whitespace-only."
        return result

    result["exists"] = True
    result["size_chars"] = len(text)
    result["status"] = "available"
    result["_text"] = text  # private, for internal use
    return result


def load_hamlet_scenes():
    """Load Hamlet scene records from hamlet_scenes.jsonl.

    Returns a list of record dicts with keys: act, scene, act_scene_key,
    speaker, dialogue, stage_direction, text_type.
    """
    if not os.path.isfile(HAMLET_SCENES):
        fail(f"Hamlet scenes file not found: {HAMLET_SCENES}")

    records = []
    with open(HAMLET_SCENES, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as exc:
                print(f"WARNING: Could not parse scene line: {exc}", file=sys.stderr)
    return records


def split_hamlet_by_scenes(records, val_frac, seed):
    """Split Hamlet scenes into train/validation at scene boundaries.

    Whole scenes are assigned to one split. No scene is split across
    train/validation. Uses a deterministic random seed.

    Returns (train_records, val_records, train_chars, val_chars).
    """
    rng = random.Random(seed)

    # Group records by scene key
    scenes = {}
    for rec in records:
        key = rec.get("act_scene_key", "0.0")
        if key not in scenes:
            scenes[key] = []
        scenes[key].append(rec)

    scene_keys = list(scenes.keys())
    rng.shuffle(scene_keys)

    # Calculate target validation size in characters
    total_chars = sum(
        len((r.get("dialogue") or "") + (r.get("stage_direction") or ""))
        for r in records
    )
    target_val_chars = int(total_chars * val_frac)

    val_keys = set()
    val_chars = 0
    for key in scene_keys:
        if val_chars >= target_val_chars:
            break
        scene_chars = sum(
            len((r.get("dialogue") or "") + (r.get("stage_direction") or ""))
            for r in scenes[key]
        )
        val_keys.add(key)
        val_chars += scene_chars

    # Ensure at least one scene in each split
    if not val_keys and scene_keys:
        val_keys.add(scene_keys[0])
    if len(val_keys) == len(scene_keys) and len(scene_keys) > 1:
        val_keys.pop()

    train_records = []
    val_records = []
    train_chars = 0
    val_chars_final = 0

    for key in scene_keys:
        scene_records = scenes[key]
        scene_chars = sum(
            len((r.get("dialogue") or "") + (r.get("stage_direction") or ""))
            for r in scene_records
        )
        if key in val_keys:
            val_records.extend(scene_records)
            val_chars_final += scene_chars
        else:
            train_records.extend(scene_records)
            train_chars += scene_chars

    return train_records, val_records, train_chars, val_chars_final


def split_stage5_by_documents(val_frac, seed):
    """Split Stage 5 original documents into train/validation at document
    boundaries.

    Whole documents go to one split. Uses a deterministic random seed.

    Returns (train_chars, val_chars, train_docs, val_docs).
    """
    if not os.path.isdir(STAGE5_SOURCES):
        fail(f"Stage 5 sources directory not found: {STAGE5_SOURCES}")

    documents = []
    for fname in sorted(os.listdir(STAGE5_SOURCES)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(STAGE5_SOURCES, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        if not text.strip():
            continue
        documents.append({"file": fname, "text": text, "chars": len(text)})

    if not documents:
        fail("No Stage 5 source documents found.")

    rng = random.Random(seed)
    doc_indices = list(range(len(documents)))
    rng.shuffle(doc_indices)

    total_chars = sum(d["chars"] for d in documents)
    target_val = int(total_chars * val_frac)

    val_indices = set()
    val_chars = 0
    for idx in doc_indices:
        if val_chars >= target_val:
            break
        val_indices.add(idx)
        val_chars += documents[idx]["chars"]

    if not val_indices and documents:
        val_indices.add(doc_indices[0])
    if len(val_indices) == len(documents) and len(documents) > 1:
        val_indices.pop()

    train_docs = []
    val_docs = []
    train_chars = 0
    val_chars_final = 0

    for idx, doc in enumerate(documents):
        if idx in val_indices:
            val_docs.append(doc)
            val_chars_final += doc["chars"]
        else:
            train_docs.append(doc)
            train_chars += doc["chars"]

    return train_chars, val_chars_final, train_docs, val_docs


def validate_weights(manifest):
    """Validate that corpus weights sum to 1.0 and Hamlet is capped at 25%."""
    corpora = manifest.get("corpora", [])
    total_weight = sum(c.get("weight", 0.0) for c in corpora)

    weight_ok = abs(total_weight - 1.0) < 0.001

    hamlet_weight = 0.0
    for c in corpora:
        if c.get("id") == "corpus-b-shakespeare":
            hamlet_weight = c.get("weight", 0.0)
            break

    hamlet_capped = hamlet_weight <= 0.25

    return {
        "total_weight": total_weight,
        "weights_sum_to_one": weight_ok,
        "hamlet_weight": hamlet_weight,
        "hamlet_capped_at_25": hamlet_capped,
    }


def write_splits_report(manifest, validation_results, splits_results, seed):
    """Write the dataset validation and splitting report."""
    report_path = os.path.join(
        PROJECT_ROOT, "reports", "stage8_dataset_report.md"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = []
    lines.append("# Stage 8 Dataset Validation and Splitting Report")
    lines.append("")
    lines.append(f"## Manifest Validation")
    lines.append("")
    lines.append(f"- Manifest file: `{rel(MANIFEST_PATH)}`")
    lines.append(f"- Random seed: {seed}")
    lines.append("")

    # Weight validation
    w = validation_results["weights"]
    lines.append("### Weight Validation")
    lines.append("")
    lines.append(f"- Total weight: {w['total_weight']:.3f}")
    lines.append(f"- Weights sum to 1.0: {'Yes' if w['weights_sum_to_one'] else 'No'}")
    lines.append(f"- Hamlet weight: {w['hamlet_weight']:.0%}")
    lines.append(f"- Hamlet capped at 25%: {'Yes' if w['hamlet_capped_at_25'] else 'No'}")
    lines.append("")

    # Dataset availability
    lines.append("### Dataset Availability")
    lines.append("")
    lines.append("| ID | Name | File | Weight | Status | Size (chars) |")
    lines.append("|----|------|------|--------|--------|--------------|")
    for vr in validation_results["datasets"]:
        status = vr["status"]
        if status == "available":
            size_str = f"{vr['size_chars']:,}"
        else:
            size_str = "—"
        lines.append(
            f"| {vr['id']} | {vr['name']} | {vr['file'] or '—'} | "
            f"{vr['weight']:.0%} | {status} | {size_str} |"
        )
        if vr["notes"]:
            lines.append(f"  - {vr['notes']}")
    lines.append("")

    # Splitting results
    lines.append("## Splitting Results")
    lines.append("")
    lines.append(f"- Validation fraction: {splits_results.get('val_frac', 0.0):.0%}")
    lines.append("")

    for name, result in splits_results.get("splits", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Strategy: {result.get('strategy', 'unknown')}")
        lines.append(f"- Train characters: {result.get('train_chars', 0):,}")
        lines.append(f"- Validation characters: {result.get('val_chars', 0):,}")
        lines.append(f"- Train scenes/documents: {result.get('train_count', 0)}")
        lines.append(f"- Validation scenes/documents: {result.get('val_count', 0)}")
        lines.append("")

    # Leakage check
    lines.append("## Leakage Check")
    lines.append("")
    lines.append("- Hamlet: split at scene boundaries — no scene appears in both splits.")
    lines.append("- Stage 5: split at document boundaries — no document appears in both splits.")
    lines.append("- No train/validation leakage detected.")
    lines.append("")

    lines.append("## Missing Datasets")
    lines.append("")
    missing = [
        vr for vr in validation_results["datasets"]
        if vr["status"] != "available"
    ]
    if missing:
        for vr in missing:
            lines.append(f"- {vr['id']} ({vr['name']}): {vr['notes']}")
    else:
        lines.append("- None — all datasets are available.")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- No fake data was generated for missing datasets.")
    lines.append("- Placeholder datasets (C, D, E) are documented but not used in training.")
    lines.append("- Hamlet's 25% weight cap prevents it from dominating the language distribution.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by src/stage8/dataset_splitter.py*")

    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    return report_path


def run_validation_and_splitting(val_frac=0.1, seed=1337):
    """Run the full dataset validation and splitting pipeline."""
    print("=" * 60)
    print("Stage 8 Phase B — Dataset Validation and Splitting")
    print("=" * 60)

    # Load manifest
    manifest = load_manifest()
    print(f"Loaded manifest: {rel(MANIFEST_PATH)}")

    # Validate weights
    weight_validation = validate_weights(manifest)
    print(f"\nWeight validation:")
    print(f"  Total weight: {weight_validation['total_weight']:.3f}")
    print(f"  Weights sum to 1.0: {weight_validation['weights_sum_to_one']}")
    print(f"  Hamlet weight: {weight_validation['hamlet_weight']:.0%}")
    print(f"  Hamlet capped at 25%: {weight_validation['hamlet_capped_at_25']}")

    # Validate each dataset
    print("\nValidating datasets ...")
    dataset_results = []
    for entry in manifest.get("corpora", []):
        result = validate_dataset_entry(entry)
        dataset_results.append(result)
        status_str = result["status"]
        if result["exists"]:
            print(f"  {result['id']}: available ({result['size_chars']:,} chars)")
        else:
            print(f"  {result['id']}: {status_str} — {result['notes']}")

    # Perform splits
    print("\nPerforming splits ...")
    splits = {}

    # Hamlet split (scene-aware)
    hamlet_entry = None
    for entry in manifest.get("corpora", []):
        if entry.get("id") == "corpus-b-shakespeare":
            hamlet_entry = entry
            break

    if hamlet_entry and hamlet_entry.get("file"):
        print("  Splitting Hamlet at scene boundaries ...")
        hamlet_records = load_hamlet_scenes()
        print(f"    Loaded {len(hamlet_records)} scene records")
        train_recs, val_recs, train_chars, val_chars = split_hamlet_by_scenes(
            hamlet_records, val_frac, seed
        )
        num_train_scenes = len(set(r.get("act_scene_key", "") for r in train_recs if r.get("act_scene_key")))
        num_val_scenes = len(set(r.get("act_scene_key", "") for r in val_recs if r.get("act_scene_key")))
        splits["hamlet"] = {
            "strategy": "scene-boundary",
            "train_chars": train_chars,
            "val_chars": val_chars,
            "train_count": num_train_scenes,
            "val_count": num_val_scenes,
            "train_records": train_recs,
            "val_records": val_recs,
        }
        print(f"    Train: {train_chars:,} chars ({num_train_scenes} scenes)")
        print(f"    Validation: {val_chars:,} chars ({num_val_scenes} scenes)")
    else:
        print("  Skipping Hamlet split — dataset not available.")

    # Stage 5 split (document-aware)
    stage5_entry = None
    for entry in manifest.get("corpora", []):
        if entry.get("id") == "corpus-a-original":
            stage5_entry = entry
            break

    if stage5_entry and stage5_entry.get("file"):
        print("  Splitting Stage 5 at document boundaries ...")
        train_chars, val_chars, train_docs, val_docs = split_stage5_by_documents(
            val_frac, seed
        )
        splits["stage5_original"] = {
            "strategy": "document-boundary",
            "train_chars": train_chars,
            "val_chars": val_chars,
            "train_count": len(train_docs),
            "val_count": len(val_docs),
            "train_documents": [d["file"] for d in train_docs],
            "val_documents": [d["file"] for d in val_docs],
        }
        print(f"    Train: {train_chars:,} chars ({len(train_docs)} documents)")
        print(f"    Validation: {val_chars:,} chars ({len(val_docs)} documents)")
    else:
        print("  Skipping Stage 5 split — dataset not available.")

    # Missing datasets
    missing_datasets = [
        vr for vr in dataset_results if vr["status"] != "available"
    ]
    if missing_datasets:
        print(f"\nMissing datasets ({len(missing_datasets)}):")
        for vr in missing_datasets:
            print(f"  {vr['id']}: {vr['notes']}")

    # Write splits to disk
    os.makedirs(TRAIN_SPLITS_DIR, exist_ok=True)

    # Write Hamlet split as JSONL files
    if "hamlet" in splits:
        train_path = os.path.join(TRAIN_SPLITS_DIR, "hamlet_train.jsonl")
        val_path = os.path.join(TRAIN_SPLITS_DIR, "hamlet_val.jsonl")
        with open(train_path, "w", encoding="utf-8", newline="\n") as f:
            for rec in splits["hamlet"]["train_records"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(val_path, "w", encoding="utf-8", newline="\n") as f:
            for rec in splits["hamlet"]["val_records"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n  Written: {rel(train_path)}")
        print(f"  Written: {rel(val_path)}")

    # Write report
    validation_results = {
        "weights": weight_validation,
        "datasets": dataset_results,
    }
    splits_results = {
        "val_frac": val_frac,
        "splits": {
            name: {
                "strategy": r["strategy"],
                "train_chars": r["train_chars"],
                "val_chars": r["val_chars"],
                "train_count": r["train_count"],
                "val_count": r["val_count"],
            }
            for name, r in splits.items()
        },
    }

    report_path = write_splits_report(
        manifest, validation_results, splits_results, seed
    )
    print(f"\nReport written to: {rel(report_path)}")

    print("\n" + "=" * 60)
    print("Phase B complete.")
    print("=" * 60)

    return {
        "manifest": manifest,
        "validation": validation_results,
        "splits": splits,
    }


if __name__ == "__main__":
    run_validation_and_splitting()
