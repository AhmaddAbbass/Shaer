import argparse
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo


def resolve_generations_repo(model_repo: str) -> str:
    if not model_repo or "/" not in model_repo:
        return ""
    owner, name = model_repo.split("/", 1)
    return f"{owner}/{name}-generations"


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_best_checkpoint_export(run_dir: Path, step: int) -> Path:
    checkpoint_dir = run_dir / f"checkpoint-{step}"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    export_dir = run_dir / f"best_checkpoint_{step}_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    keep_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "grpo_resume_manifest.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "trainer_state.json",
        "training_args.bin",
    ]
    for name in keep_files:
        src = checkpoint_dir / name
        if src.exists():
            shutil.copy2(src, export_dir / name)

    summary = read_json(run_dir / "final_plots" / "best_checkpoint_summary.json") or {}
    readme_lines = [
        f"# Best Checkpoint Export ({step})",
        "",
        f"- source run: `{run_dir.name}`",
        f"- source checkpoint: `checkpoint-{step}`",
    ]
    if summary:
        readme_lines.extend(
            [
                f"- best eval total: `{float(summary.get('best_eval_total', 0.0)):.4f}`",
                f"- best eval meter: `{float(summary.get('best_eval_meter', 0.0)):.4f}`",
                f"- best eval count adherence: `{float(summary.get('best_eval_count_adherence', 0.0)):.4f}`",
                f"- best eval repeat penalty: `{float(summary.get('best_eval_repeat_penalty', 0.0)):.4f}`",
                f"- best eval arabic clean: `{float(summary.get('best_eval_arabic_clean', 0.0)):.4f}`",
            ]
        )
    readme_lines.extend(
        [
            "",
            "This export keeps the inference-relevant best-checkpoint files plus trainer metadata.",
            "Optimizer and scheduler state are intentionally excluded from this paper-facing Hub export.",
            "",
        ]
    )
    (export_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    return export_dir


def main():
    parser = argparse.ArgumentParser(description="Upload paper-ready GRPO artifacts and best checkpoint to Hugging Face Hub.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-repo", default="")
    parser.add_argument("--generations-repo", default="")
    parser.add_argument("--paper-dir", default="")
    parser.add_argument("--best-step", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found")

    run_dir = Path(args.run_dir).resolve()
    effective = read_json(run_dir / "effective_runtime_config.json") or {}
    model_repo = (
        str(args.model_repo).strip()
        or str(effective.get("effective_trainer", {}).get("hub_model_id", "")).strip()
        or str(effective.get("run", {}).get("output_repo", "")).strip()
    )
    generations_repo = (
        str(args.generations_repo).strip()
        or str(effective.get("effective_trainer", {}).get("generations_repo_id", "")).strip()
        or resolve_generations_repo(model_repo)
    )
    paper_dir = Path(args.paper_dir).resolve() if str(args.paper_dir).strip() else run_dir / "final_plots"
    best_step = int(args.best_step) if int(args.best_step or 0) > 0 else int((read_json(run_dir / "final_plots" / "best_checkpoint_summary.json") or {}).get("best_step", 0))

    if not model_repo:
        raise RuntimeError("Could not resolve model repo")
    if not generations_repo:
        raise RuntimeError("Could not resolve generations repo")
    if not paper_dir.exists():
        raise FileNotFoundError(f"Paper dir not found: {paper_dir}")
    if best_step <= 0:
        raise RuntimeError("Could not resolve best checkpoint step")

    api = HfApi(token=hf_token)
    create_repo(repo_id=model_repo, repo_type="model", token=hf_token, exist_ok=True)
    create_repo(repo_id=generations_repo, repo_type="dataset", token=hf_token, exist_ok=True)

    best_export_dir = prepare_best_checkpoint_export(run_dir, best_step)

    readme_path = run_dir / "README.md"
    if readme_path.exists():
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=model_repo,
            repo_type="model",
            token=hf_token,
            commit_message=f"Update model card for {run_dir.name}",
        )

    api.upload_folder(
        folder_path=str(paper_dir),
        path_in_repo="paper/final_plots",
        repo_id=model_repo,
        repo_type="model",
        token=hf_token,
        commit_message=f"Upload paper plots for {run_dir.name}",
    )
    api.upload_folder(
        folder_path=str(best_export_dir),
        path_in_repo="best-checkpoint",
        repo_id=model_repo,
        repo_type="model",
        token=hf_token,
        commit_message=f"Upload best checkpoint export for {run_dir.name}",
    )
    api.upload_folder(
        folder_path=str(paper_dir),
        path_in_repo=f"runs/{run_dir.name}/final_plots",
        repo_id=generations_repo,
        repo_type="dataset",
        token=hf_token,
        commit_message=f"Upload paper bundle for {run_dir.name}",
    )

    print(
        json.dumps(
            {
                "model_repo": model_repo,
                "generations_repo": generations_repo,
                "best_step": best_step,
                "paper_dir": str(paper_dir),
                "best_export_dir": str(best_export_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
