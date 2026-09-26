"""Download the pinned Cauldron parquet shards for the probe subsets (parallel, resumable)."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from cauldron_tasks import CAULDRON, CAULDRON_REVISION, SUBSETS

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path.home() / "cauldron")
ap.add_argument("--subsets", nargs="+", default=SUBSETS)
ap.add_argument("--workers", type=int, default=16)
args = ap.parse_args()
snapshot_download(CAULDRON, repo_type="dataset", revision=CAULDRON_REVISION, local_dir=args.out,
                  allow_patterns=[f"{s}/*.parquet" for s in args.subsets], max_workers=args.workers)
print("downloaded to", args.out)
