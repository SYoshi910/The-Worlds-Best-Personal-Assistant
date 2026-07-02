"""Offline BGE-M3 embeddings for task title matching."""

import os

# Block Hugging Face Hub before any transformers import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path

from config import EMBEDDING_DEVICE, EMBEDDING_MODEL_PATH

_model = None


def _model_dir() -> Path:
    return Path(EMBEDDING_MODEL_PATH)


def _get_embed_model():
    global _model
    if _model is not None:
        return _model

    path = _model_dir()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Embedding model not found at {path}. "
            "Run: python scripts/download_bge_m3.py"
        )

    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer(
        str(path),
        device=EMBEDDING_DEVICE,
        model_kwargs={"local_files_only": True},
        processor_kwargs={"local_files_only": True},
    )
    return _model


def warmup_embeddings() -> None:
    _get_embed_model()


def embed_text(text: str) -> list[float]:
    vec = _get_embed_model().encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_documents(titles: list[str]) -> list[list[float]]:
    if not titles:
        return []
    vecs = _get_embed_model().encode(
        titles, batch_size=32, normalize_embeddings=True
    )
    return vecs.tolist()
