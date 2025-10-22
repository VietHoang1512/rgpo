#!/usr/bin/env python3
"""
inspect_reasoning_domains.py

Walk the downloaded reasoning subsets and print a concise summary for each:
- file types and counts
- inferred splits (from filenames)
- schema keys (from sample records)
- N sample records (pretty-printed & truncated)
- image stats (local files, URL fields)

Usage:
  python3 inspect_reasoning_domains.py \
      --data-root /path/to/reasoning_data \
      --samples 2 --per-file-samples 2

Tips:
- If your downloader module is in the same directory, the script will
  import REASONING_DATASETS; otherwise it inspects all subfolders.
"""

import argparse
import json
import os
import re
import sys
import csv
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# Try to reuse your curated list, fall back to "all subdirs"
REASONING_DATASETS = [
    # Diagram and geometry reasoning
    "ai2d",                     # procedural/part–whole diagram QA【698328341205276†L64-L108】
    "CLEVR",                   # synthetic compositional reasoning【698328341205276†L64-L72】
    "CLEVR-Math",              # CLEVR variant with math word problems【698328341205276†L64-L68】
    "Super-CLEVR",             # more complex CLEVR scenes【698328341205276†L98-L100】
    "GeoQA+",                  # geometry question answering【698328341205276†L84-L87】
    "Geometry3K",              # geometry diagrams and reasoning【698328341205276†L86-L89】
    "GEOS",                    # geometry problem solving【698328341205276†L81-L83】
    "geo170k_qa",              # large-scale geometry QA
    "geo3k",                   # small geometry dataset【698328341205276†L186-L188】
    "IconQA",                  # icon‑based diagram reasoning【698328341205276†L88-L92】
    # "raven",                   # Raven’s Progressive Matrices (visual IQ)【428668411446964†screenshot】
    "unigeo",                  # UniGeo: geometry question answering【410243429803351†screenshot】

    # Chart, plot and table reasoning
    "chartqa",                 # chart question answering【698328341205276†L147-L152】
    # "chart2text",             # textualisation of charts【698328341205276†L146-L149】
    "dvqa",                    # bar chart reasoning【698328341205276†L171-L176】
    "FigureQA",                # synthetic figure reasoning【698328341205276†L78-L80】
    "plotqa",                  # plot reasoning (line plots etc.)【428668411446964†screenshot】
    "tallyqa",                 # counting objects in complex scenes【410243429803351†screenshot】
    "tabmwp",                  # table-based math word problems【410243429803351†screenshot】
    # "finqa",                   # financial table QA【698328341205276†L174-L178】
    # "hitab",                   # hybrid table QA【698328341205276†L204-L208】
    # "tat_qa",                  # Tabular QA with arithmetic reasoning【410243429803351†screenshot】
    # "tinychart_train",         # tiny chart QA dataset【410243429803351†screenshot】
    # "tqa",                     # Textbook question answering (science diagrams)【410243429803351†screenshot】
    # "lrv_chart",               # chart QA distilled from GPT‑4V【335508432603398†screenshot】
    "infographic_vqa",         # infographic reasoning【335508432603398†screenshot】
    # "infographic_azuregpt4v",  # infographic reasoning via GPT‑4V【335508432603398†screenshot】
    # "mapqa",                   # map question answering【335508432603398†screenshot】

    # Document and text‑rich QA (reading + reasoning)
    # "docvqa_train",            # general document QA【698328341205276†L170-L173】
    # "OmniDocBench_train",      # benchmark of diverse doc tasks【698328341205276†L92-L95】
    # "uReader_cap",             # UReader captioning (doc images)
    # "uReader_chart",           # UReader chart QA【410243429803351†screenshot】
    # "uReader_ie",              # UReader information extraction【410243429803351†screenshot】
    # "uReader_kg",              # UReader knowledge graph reasoning【410243429803351†screenshot】
    # "uReader_ocr",             # UReader OCR tasks【410243429803351†screenshot】
    # "uReader_qa",              # UReader question answering【810401881967551†screenshot】
    # "uReader_tr",              # UReader translation【410243429803351†screenshot】
    # "ST_VQA",                  # scene‑text VQA【410243429803351†screenshot】
    # "textvqa",                 # reading text in the wild【410243429803351†screenshot】
    # "textcaps",                # captioning of text‑rich images【410243429803351†screenshot】
    # "textocr_gpt4v",           # OCR with GPT‑4V【410243429803351†screenshot】
    # "ocrvqa",                  # OCR question answering【428668411446964†screenshot】
    # "pathvqa",                 # pathology image QA【428668411446964†screenshot】
    # "PMC-VQA",                 # biomedical QA【698328341205276†L94-L97】
    # "sroie_data",              # receipt OCR and QA【410243429803351†screenshot】
    # "screen2words",            # UI screenshot captioning【428668411446964†screenshot】
    # "screen_qa",               # UI screenshot QA【428668411446964†screenshot】
    # "VisualWebInstruct",       # web UI QA (requires reasoning)【698328341205276†L100-L104】

    # General/knowledge VQA and science
    "gqa",                     # compositional general‑knowledge VQA【698328341205276†L198-L200】
    "aokvqa",                  # knowledge‑intensive VQA【698328341205276†L132-L134】
    "scienceqa",               # science question answering with images【428668411446964†screenshot】
    # "vqaas",                   # VQA assistant style【810401881967551†screenshot】
    # "vqarad",                  # radiology VQA (health)【810401881967551†screenshot】

    # Additional synthetic and analytical reasoning tasks
    # "RAVEN",                   # Raven’s Progressive Matrices【428668411446964†screenshot】
    "synthetic_amc",           # AMC‑style synthetic math problems【410243429803351†screenshot】
    "synthetic_math",          # synthetic math QA【410243429803351†screenshot】
]


DATA_FILE_EXTS = {".jsonl", ".json", ".csv", ".tsv", ".parquet"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
SPLIT_HINTS = ("train", "val", "valid", "validation", "dev", "test")


def shorten(x, max_len=240):
    """Shorten long strings and containers for printing."""
    if isinstance(x, str):
        x = x.replace("\n", "\\n")
        return (x[:max_len] + "…") if len(x) > max_len else x
    if isinstance(x, (list, tuple)):
        return [shorten(v, max_len=max_len // 2) for v in x[:5]]
    if isinstance(x, dict):
        out = {}
        for i, (k, v) in enumerate(list(x.items())[:10]):
            out[str(k)] = shorten(v, max_len=max_len // 2)
        return out
    return x


def detect_image_fields(rec):
    """Heuristically detect image fields/urls in a record."""
    keys = set(rec.keys()) if isinstance(rec, dict) else set()
    probable = []
    for k in keys:
        if any(t in k.lower() for t in ["image", "img", "picture", "figure"]):
            probable.append(k)
    return probable


def infer_split_from_name(name: str):
    name_low = name.lower()
    for s in SPLIT_HINTS:
        if s in name_low:
            return s
    return None


def iter_jsonl(fp, limit=5):
    """Yield up to `limit` JSONL rows from file path `fp`."""
    count = 0
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj
                count += 1
                if count >= limit:
                    break
            except Exception:
                continue


def load_json_sample(fp, limit=5):
    """Return up to `limit` samples from a .json file (list or dict)."""
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
    except Exception as e:
        return []
    if isinstance(obj, list):
        return obj[:limit]
    if isinstance(obj, dict):
        # common wrappers: {"data":[...]} or {"train":[...], "val":[...]}
        for k in ("data", "train", "validation", "val", "test", "examples", "items"):
            if k in obj and isinstance(obj[k], list):
                return obj[k][:limit]
        # fallback: show up to 'limit' key/value pairs
        return [{k: obj[k]} for i, k in enumerate(obj.keys()) if i < limit]
    return []


def sample_csv_tsv(fp, limit=5):
    if pd is None:
        # light fallback with csv module
        rows, keys = [], None
        with open(fp, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter="\t" if fp.endswith(".tsv") else ",")
            for i, row in enumerate(reader):
                rows.append(row)
                if i + 1 >= limit:
                    break
        return rows
    try:
        df = pd.read_csv(fp, sep="\t" if fp.endswith(".tsv") else ",", nrows=limit)
        return df.to_dict(orient="records")
    except Exception:
        return []


def sample_parquet(fp, limit=5):
    if pd is None:
        return []
    try:
        df = pd.read_parquet(fp)
        return df.head(limit).to_dict(orient="records")
    except Exception:
        return []


def summarize_dataset(ds_dir: Path, per_file_samples: int):
    print(f"\n=== DATASET: {ds_dir.name}")
    # file inventory
    files = [p for p in ds_dir.rglob("*") if p.is_file()]
    if not files:
        print("  (empty)")
        return

    ext_count = Counter(p.suffix.lower() for p in files)
    print("  file types:", dict(sorted(ext_count.items(), key=lambda kv: (-kv[1], kv[0]))))

    # split hints
    splits = set()
    for p in files:
        s = infer_split_from_name(p.name)
        if s:
            splits.add(s)
    if splits:
        print("  inferred splits:", ", ".join(sorted(splits)))

    # image stats
    local_imgs = [p for p in files if p.suffix.lower() in IMG_EXTS]
    print(f"  local images: {len(local_imgs)}")

    # candidate data files
    data_files = [p for p in files if p.suffix.lower() in DATA_FILE_EXTS]
    if not data_files:
        print("  data files: (none found)")
        return

    print(f"  data files found: {len(data_files)}")

    # quick schema + samples
    for p in sorted(data_files):
        print(f"\n  -> {p.relative_to(ds_dir)}")
        print(f"     size: {p.stat().st_size/1024:.1f} KB")
        samples = []
        if p.suffix.lower() == ".jsonl":
            samples = list(iter_jsonl(p, limit=per_file_samples))
        elif p.suffix.lower() == ".json":
            samples = load_json_sample(p, limit=per_file_samples)
        elif p.suffix.lower() in (".csv", ".tsv"):
            samples = sample_csv_tsv(p, limit=per_file_samples)
        elif p.suffix.lower() == ".parquet":
            samples = sample_parquet(p, limit=per_file_samples)

        # collect keys from samples
        key_counter = Counter()
        for s in samples:
            if isinstance(s, dict):
                key_counter.update(s.keys())
        if key_counter:
            top_keys = [k for k, _ in key_counter.most_common(12)]
            print("     keys (sample):", top_keys)

        # detect probable image fields
        if samples:
            img_fields = set()
            for s in samples:
                if isinstance(s, dict):
                    for k in detect_image_fields(s):
                        img_fields.add(k)
            if img_fields:
                print("     probable image fields:", sorted(img_fields))

        # show samples
        for i, s in enumerate(samples):
            s_disp = shorten(s, max_len=240)
            # Pretty print but compact
            as_json = json.dumps(s_disp, ensure_ascii=False)
            print(f"     sample[{i}]: {as_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="Path to the folder where you downloaded the subsets (the --output-dir you used).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Limit to these dataset folder names (default: use REASONING_DATASETS or all subdirs).")
    ap.add_argument("--samples", type=int, default=2,
                    help="How many datasets to sample across (0 = all).")
    ap.add_argument("--per-file-samples", type=int, default=2,
                    help="How many rows to show from each data file.")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    if not root.exists():
        print(f"Data root not found: {root}", file=sys.stderr)
        sys.exit(1)

    # Decide which dataset dirs to walk
    if args.datasets:
        ds_names = args.datasets
    elif REASONING_DATASETS:
        ds_names = [d for d in REASONING_DATASETS if (root / d).exists()]
    else:
        ds_names = [p.name for p in root.iterdir() if p.is_dir()]

    if not ds_names:
        print("No dataset subfolders found under:", root)
        sys.exit(0)

    ds_names = sorted(ds_names)
    if args.samples and args.samples > 0:
        ds_names = ds_names[: args.samples]

    print(f"Inspecting {len(ds_names)} subset(s) under {root}")
    for name in ds_names:
        ds_dir = root / name
        if not ds_dir.exists():
            print(f"\n=== DATASET: {name}\n  (missing at {ds_dir})")
            continue
        summarize_dataset(ds_dir, per_file_samples=args.per_file_samples)


if __name__ == "__main__":
    main()
