"""Verify embeddings load and run with no network access."""

import socket
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embeddings
from config import EMBEDDING_MODEL_PATH


def _block_network():
    return patch(
        "socket.socket",
        side_effect=OSError("network disabled for offline test"),
    )


def main() -> None:
    path = Path(EMBEDDING_MODEL_PATH)
    if not (path / "config.json").is_file():
        print(f"Skip: model not at {path}. Run scripts/download_bge_m3.py first.")
        sys.exit(1)

    embeddings._model = None
    with _block_network():
        vec = embeddings.embed_text("bcg prep")
        docs = embeddings.embed_documents(["BCG prep", "daily briefing"])

    assert len(vec) == 1024, f"expected 1024-dim vector, got {len(vec)}"
    assert len(docs) == 2 and len(docs[0]) == 1024
    print("Offline embedding test passed (no network).")


if __name__ == "__main__":
    main()
