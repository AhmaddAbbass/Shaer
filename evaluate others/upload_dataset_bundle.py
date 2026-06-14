#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a prepared evaluation bundle folder to a Hugging Face dataset repository.")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token-env-var", default="HF_TOKEN")
    parser.add_argument("--commit-message", default="Upload evaluation bundle")
    parser.add_argument("--path-in-repo", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv(args.token_env_var) or ""
    if not token:
        raise RuntimeError(f"Missing Hugging Face token in environment variable {args.token_env_var}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=bool(args.private), exist_ok=True)
    api.upload_folder(
        folder_path=str(Path(args.folder)),
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo=str(args.path_in_repo or ""),
        commit_message=str(args.commit_message or "Upload evaluation bundle"),
    )
    print(f"uploaded_folder={args.folder} repo_id={args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
