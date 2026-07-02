"""One-time download of BGE-M3 into models/bge-m3/ (requires network)."""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "models" / "bge-m3"
REPO = "BAAI/bge-m3"


def main() -> None:
    if (TARGET / "config.json").is_file():
        print(f"Model already present at {TARGET}")
        return

    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO} → {TARGET} (~2.2 GB, one-time)...")
    snapshot_download(
        repo_id=REPO,
        local_dir=str(TARGET),
        local_dir_use_symlinks=False,
    )
    print("Done. Embeddings will run fully offline.")


if __name__ == "__main__":
    main()
