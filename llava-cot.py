"""
download_llava_cot.py
---------------------

This script automates the process of retrieving the *LLaVA-CoT‑100k* dataset
from the Hugging Face Hub, rebuilding the split image archive and extracting
its contents.  The dataset is hosted on the Hugging Face platform under
``Xkev/LLaVA-CoT-100k`` and contains a large number of images split across
multiple ``image.zip.part-*`` files, along with a ``train.jsonl`` metadata
file.  According to the dataset card, the parts must be concatenated into
a single ``image.zip`` archive before unzipping【569876990342035†L2898-L2907】.

The script relies on the ``huggingface_hub`` library to download the
individual parts and the JSONL file.  After downloading all parts, it
merges them sequentially, writes the merged archive to disk, and then
extracts the images using Python's built‑in ``zipfile`` module.  You can
configure the output directory where images will be extracted by
setting the ``--output-dir`` argument.

Usage
-----

Run this script from a terminal with Python 3 installed.  If you haven't
installed ``huggingface_hub`` yet, first run:

    pip install huggingface_hub

Then execute the script:

    python download_llava_cot.py --output-dir ./llava_images

The script will download each ``image.zip.part-aa`` through
``image.zip.part-ap`` file from the Hub, combine them into ``image.zip``,
unzip the archive into ``./llava_images``, and also download the
``train.jsonl`` file to the current working directory.  Keep in mind that
the combined image archive is roughly 171 GB in size, so the download and
extraction will take considerable time and disk space.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path




# def download_file(repo_id: str, filename: str, repo_type: str = "dataset", token: str | None = None) -> Path:
#     """Download a file from the Hugging Face Hub and return its local path.

#     Parameters
#     ----------
#     repo_id: str
#         The repository identifier on the Hugging Face Hub (e.g., ``Xkev/LLaVA-CoT-100k``).
#     filename: str
#         The filename within the repository to download.
#     repo_type: str, optional
#         The type of repository; use ``"dataset"`` for datasets.  Defaults to ``"dataset"``.
#     token: str or None, optional
#         Authentication token for private datasets.  If the dataset is public,
#         this can remain ``None``.

#     Returns
#     -------
#     Path
#         The path to the downloaded file in the local Hugging Face cache.
#     """
#     return Path(
#         hf_hub_download(
#             repo_id=repo_id,
#             filename=filename,
#             repo_type=repo_type,
#             token=token,
#         )
#     )


def merge_parts(part_paths, output_path):
    """Concatenate multiple part files into a single archive.

    Parameters
    ----------
    part_paths: list[Path]
        A list of paths to the part files in the order they should be concatenated.
    output_path: Path
        The path where the combined archive should be written.
    """
    with output_path.open("wb") as outfile:
        for part in part_paths:
            with part.open("rb") as infile:
                while True:
                    chunk = infile.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    outfile.write(chunk)


def extract_zip(zip_path, extract_dir) :
    """Extract a ZIP archive to the specified directory.

    Parameters
    ----------
    zip_path: Path
        The path to the ZIP archive.
    extract_dir: Path
        The directory where files should be extracted.  The directory
        is created if it doesn't already exist.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=extract_dir)


def download_and_prepare_dataset(output_dir, token) :
    """Download LLaVA-CoT‑100k image parts and metadata, merge and extract them.

    Parameters
    ----------
    output_dir: Path
        The directory into which images from the reconstructed ZIP archive
        should be extracted.
    token: str or None, optional
        Authentication token for private datasets.  Not required for this
        public dataset.
    """
    repo_id = "Xkev/LLaVA-CoT-100k"
    repo_type = "dataset"

    # Generate part filenames from aa to ap (inclusive).
    suffixes = [chr(code) for code in range(ord("a"), ord("p") + 1)]
    part_filenames = [f"image.zip.part-a{suffix}" for suffix in suffixes]

    # Download all parts in order.
    part_paths: list[Path] = []
    for fname in part_filenames:
        print(f"Downloading {fname}...")
        # part_path = download_file(repo_id=repo_id, filename=fname, repo_type=repo_type, token=token)
        part_path = Path("/vast/hvp2011/data/LLaVA-CoT-100k/" + fname)
        part_paths.append(part_path)
    print("All parts downloaded.")

    # Download metadata file
    # print("Downloading train.jsonl...")
    # train_json_path = download_file(repo_id=repo_id, filename="train.jsonl", repo_type=repo_type, token=token)
    # print(f"train.jsonl downloaded to {train_json_path}")

    # Merge parts into one ZIP file.
    merged_zip_path = Path("image.zip")
    print(f"Merging parts into {merged_zip_path}...")
    merge_parts(part_paths, merged_zip_path)
    print(f"Created merged archive {merged_zip_path}")

    # Extract the merged ZIP archive.
    print(f"Extracting {merged_zip_path} to {output_dir}...")
    extract_zip(merged_zip_path, output_dir)
    print("Extraction complete.")


def main() -> int:
    """Parse command‑line arguments and download the dataset accordingly."""
    parser = argparse.ArgumentParser(
        description="Download and extract the LLaVA-CoT‑100k dataset images."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/vast/hvp2011/data/LLaVA-CoT-100k/llava_cot_images"),
        help="Directory to extract images into. Defaults to /vast/hvp2011/data/LLaVA-CoT-100k/llava_cot_images."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face authentication token if required."
    )
    args = parser.parse_args()
    download_and_prepare_dataset(output_dir=args.output_dir, token=args.token)
    return 0


if __name__ == "__main__":
    main()