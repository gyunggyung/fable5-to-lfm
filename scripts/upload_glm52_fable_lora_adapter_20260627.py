#!/usr/bin/env python3
"""Upload GLM-5.2 Fable LoRA adapter artifacts to Hugging Face.

This intentionally uploads adapter-sized artifacts and metadata, not the full
743B base checkpoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, upload_folder
from huggingface_hub.errors import HfHubHTTPError


DEFAULT_PATTERNS = [
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "training_args.bin",
    "trainer_state.json",
    "args.json",
    "latest_checkpointed_iteration.txt",
    "**/adapter_config.json",
    "**/adapter_model.safetensors",
    "**/adapter_model.bin",
    "**/trainer_state.json",
    "**/args.json",
    "**/latest_checkpointed_iteration.txt",
    "**/README.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        default="/home/work/.data/harness1/models/zai-org__GLM-5.2__GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA-20260627",
        help="Training output or checkpoint folder to upload from.",
    )
    parser.add_argument(
        "--repo-id",
        default="LLM-OS-Models/GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA",
    )
    parser.add_argument(
        "--readme",
        default="/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/model_cards/GLM-5.2-FP8-Agentic-Fable5-Composer2.5-LoRA-README.md",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Upload GLM-5.2 Fable LoRA adapter checkpoint")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Upload only the model card when the training folder does not exist yet.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).expanduser().resolve()
    readme = Path(args.readme).expanduser().resolve()

    if not folder.exists() and not args.allow_empty:
        raise SystemExit(f"missing upload folder: {folder}")
    if not readme.exists():
        raise SystemExit(f"missing model card: {readme}")

    adapter_files = []
    if folder.exists():
        adapter_files = [
            p
            for p in folder.rglob("*")
            if p.name in {"adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"}
        ]
    if not adapter_files:
        print("warning: no adapter_model/adapter_config files found yet; uploading metadata only")

    print(f"repo_id={args.repo_id}")
    print(f"folder={folder}")
    print(f"readme={readme}")
    print(f"adapter_files={len(adapter_files)}")
    if args.dry_run:
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        raise SystemExit(
            "Hugging Face upload auth failed while creating the repo. "
            "Set a valid HF_TOKEN or HUGGINGFACE_HUB_TOKEN with write access."
        ) from exc

    try:
        with TemporaryDirectory(prefix="glm52-fable-upload-") as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "README.md").write_text(readme.read_text(encoding="utf-8"), encoding="utf-8")
            upload_folder(
                folder_path=str(tmp),
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                commit_message="Update GLM-5.2 Fable LoRA model card",
                token=token,
            )

        if folder.exists():
            upload_folder(
                folder_path=str(folder),
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                allow_patterns=DEFAULT_PATTERNS,
                commit_message=args.commit_message,
                token=token,
            )
    except HfHubHTTPError as exc:
        raise SystemExit(
            "Hugging Face upload failed. Check repo permissions, network access, "
            "and the HF_TOKEN/HUGGINGFACE_HUB_TOKEN value."
        ) from exc


if __name__ == "__main__":
    main()
