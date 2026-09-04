from typing import TYPE_CHECKING, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from ..config.settings import AppSettings


class Embedder(Protocol):
    """Turns text into a query/index vector.

    Shared seam between the query path and the embedding/upload pipeline, so
    both produce vectors with the same model. `embed_many` is the batch entry
    point the bulk uploader uses; `embed` is the single-text convenience.
    """

    def embed(self, text: str) -> np.ndarray: ...

    def embed_many(self, texts: list[str]) -> list[np.ndarray]: ...


def _resolve_device(preferred: str) -> str:
    """Pick a torch device: `auto` chooses CUDA > MPS > CPU."""
    import torch

    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if preferred == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("EMBED_DEVICE=cuda but torch reports no CUDA device")
    if preferred == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("EMBED_DEVICE=mps but torch reports no MPS device")
    return preferred


class SentenceTransformerEmbedder:
    """Embedder backed by sentence-transformers (torch).

    Device selection is `auto` by default: CUDA if present, else MPS on Apple
    Silicon, else CPU. Set `device` explicitly (`cpu`/`cuda`/`mps`) to pin.
    Vectors are L2-normalized so cosine similarity is a plain dot product,
    matching what the Qdrant collection is configured for.

    `batch_size` is the encode-time batch handed to torch. `threads` only
    applies on CPU (it caps torch's intra-op threading via
    `torch.set_num_threads`); GPU devices ignore it.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 256,
        threads: int | None = None,
    ):
        resolved = _resolve_device(device)
        if resolved == "cpu" and threads is not None:
            import torch

            torch.set_num_threads(threads)
        self._model = SentenceTransformer(model_name, device=resolved)
        self.device = resolved
        self.batch_size = batch_size
        self.threads = threads

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        return [vectors[i] for i in range(vectors.shape[0])]


def make_embedder(settings: "AppSettings") -> Embedder:
    """Construct the configured embedder.

    The single place backend selection would branch if a non-sentence-
    transformers model is ever needed.
    """
    return SentenceTransformerEmbedder(
        settings.model_name,
        device=settings.embed_device,
        batch_size=settings.embed_batch_size,
        threads=settings.embed_threads,
    )
