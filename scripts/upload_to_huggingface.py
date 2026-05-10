"""Upload the turbofan-surrogate release to HuggingFace Hub.

Uses an *allow-list* (whitelist) of files to push. Only the files listed
in ALLOW_PATTERNS are uploaded — anything else (.venv, .git, *.pyc,
build artefacts, etc.) is automatically excluded by construction. This
is safer than relying on ignore_patterns, which fnmatch interprets too
narrowly for nested directories.

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

Preview which files would upload with --dry-run before committing.
"""
from __future__ import annotations

import argparse
import os
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Whitelist — anything not matching one of these globs is left out.
# Matched against POSIX-style paths relative to the release dir root.
# ---------------------------------------------------------------------------
ALLOW_PATTERNS: List[str] = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    ".gitignore",
    "weights/*.pkl",
    "reports/*.md",
    "examples/*.py",
    "docs/*.md",
    "turbofan_surrogate/__init__.py",
    "turbofan_surrogate/constants.py",
    "turbofan_surrogate/inference.py",
    "turbofan_surrogate/model.py",
    "turbofan_surrogate/simulator.py",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="LucasThil/turbofan-surrogate",
                   help="HuggingFace model repo (creates if missing).")
    p.add_argument("--commit-message", default="Update turbofan-surrogate release",
                   help="Commit message on the Hub.")
    p.add_argument("--release-dir", default=".",
                   help="Path to the release directory root (defaults to cwd).")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF token; defaults to HF_TOKEN env var.")
    p.add_argument("--private", action="store_true",
                   help="Create the repo as private.")
    p.add_argument("--dry-run", action="store_true",
                   help="List files that would be uploaded; do not call Hub.")
    return p.parse_args()


def _enumerate_allowed_files(release: Path) -> List[Path]:
    """Walk the release dir and keep only files matching ALLOW_PATTERNS."""
    keep: List[Path] = []
    for p in release.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(release).as_posix()
        if any(fnmatch(rel, pat) for pat in ALLOW_PATTERNS):
            keep.append(p)
    return sorted(keep)


def main() -> None:
    args = parse_args()

    release = Path(args.release_dir).resolve()
    if not (release / "README.md").exists():
        print(f"ERROR: {release} does not look like a release dir "
              "(missing README.md).", file=sys.stderr)
        sys.exit(1)

    print(f"Release dir : {release}")
    print(f"Repo id     : {args.repo_id}")
    print(f"\nAllow patterns:")
    for pat in ALLOW_PATTERNS:
        print(f"  {pat}")

    keep = _enumerate_allowed_files(release)
    total_kb = sum(p.stat().st_size for p in keep) / 1024
    print(f"\nFiles selected for upload: {len(keep)} ({total_kb:.1f} KB total)")
    for p in keep:
        rel = p.relative_to(release).as_posix()
        size = p.stat().st_size
        if size > 1024 * 1024:
            print(f"  {rel}  ({size / (1024*1024):.2f} MB)")
        else:
            print(f"  {rel}  ({size / 1024:.1f} KB)")

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

    print("Uploading (whitelisted files only) ...")
    api.upload_folder(
        folder_path    = str(release),
        repo_id        = args.repo_id,
        repo_type      = "model",
        commit_message = args.commit_message,
        allow_patterns = ALLOW_PATTERNS,    # whitelist; everything else dropped
        token          = args.token,
    )
    print(f"\nDone. https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
