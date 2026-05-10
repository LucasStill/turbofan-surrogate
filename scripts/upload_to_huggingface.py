"""Upload the turbofan-surrogate release to HuggingFace Hub.

Run this on your laptop with your HuggingFace token in the environment.
Do not commit the token. The script uploads weights/, reports/, the
model card, README, LICENSE, and pyproject metadata to a model repo.

Setup
-----
    pip install huggingface_hub
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
    # or `huggingface-cli login`

Usage
-----
    python scripts/upload_to_huggingface.py \
        --repo-id LucasThil/turbofan-surrogate \
        --commit-message "Initial release v0.1.0"

You can preview which files would upload with --dry-run.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="LucasThil/turbofan-surrogate",
                   help="HuggingFace model repo (creates if missing).")
    p.add_argument("--commit-message", default="Update turbofan-surrogate release",
                   help="Commit message on the Hub.")
    p.add_argument("--release-dir", default=".",
                   help="Path to the release directory root "
                        "(defaults to current dir).")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF token; defaults to HF_TOKEN env var.")
    p.add_argument("--private", action="store_true",
                   help="Create the repo as private.")
    p.add_argument("--dry-run", action="store_true",
                   help="List files that would be uploaded; do not call Hub.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    release = Path(args.release_dir).resolve()
    if not (release / "README.md").exists():
        print(f"ERROR: {release} does not look like a release dir "
              "(missing README.md).", file=sys.stderr)
        sys.exit(1)

    # Files / folders to upload
    paths_to_upload = [
        release / "README.md",
        release / "LICENSE",
        release / "pyproject.toml",
        release / "weights",
        release / "reports",
        release / "examples",
        release / "docs",
        release / "turbofan_surrogate",
    ]
    paths_to_upload = [p for p in paths_to_upload if p.exists()]

    print(f"Release dir: {release}")
    print(f"Repo id    : {args.repo_id}")
    print("Files to upload:")
    for p in paths_to_upload:
        if p.is_dir():
            for sub in sorted(p.rglob("*")):
                if sub.is_file() and "__pycache__" not in str(sub):
                    rel = sub.relative_to(release)
                    print(f"  {rel}  ({sub.stat().st_size / 1024:.1f} KB)")
        else:
            rel = p.relative_to(release)
            print(f"  {rel}  ({p.stat().st_size / 1024:.1f} KB)")

    if args.dry_run:
        print("\n[dry run] not uploading.")
        return

    if not args.token:
        print("ERROR: no HF token (set HF_TOKEN env var or pass --token).",
              file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub not installed.\n"
              "       pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=args.token)
    print(f"\nCreating / fetching repo {args.repo_id} ...")
    create_repo(args.repo_id, token=args.token, exist_ok=True,
                private=args.private, repo_type="model")

    print("Uploading ...")
    # Single multi-file commit
    api.upload_folder(
        folder_path     = str(release),
        repo_id         = args.repo_id,
        repo_type       = "model",
        commit_message  = args.commit_message,
        ignore_patterns = ["__pycache__", "*.pyc", ".venv", "build", "dist",
                           ".git", ".DS_Store", "scripts/upload_to_huggingface.py"],
        token           = args.token,
    )
    print(f"\nDone. https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
