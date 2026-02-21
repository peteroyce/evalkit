"""SemanticSimilarityScorer — cosine similarity of text representations."""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

from evalkit.core.types import Score
from evalkit.scorers.base import BaseScorer

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Compute a TF-IDF vector from a list of tokens and a precomputed IDF map."""
    tf = Counter(tokens)
    total = max(len(tokens), 1)
    return {t: (count / total) * idf.get(t, 1.0) for t, count in tf.items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors (dict form)."""
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in a)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticSimilarityScorer(BaseScorer):
    """Scores outputs using cosine similarity of text representations.

    Tries to use ``sentence-transformers`` for dense embeddings when available.
    Falls back to a TF-IDF cosine similarity implementation if the library is
    not installed (no external dependencies required for basic operation).

    Args:
        model_name: Sentence-transformers model to use (default: all-MiniLM-L6-v2).
        use_tfidf_fallback: If True and sentence-transformers is unavailable,
            use the built-in TF-IDF fallback. If False, raises ImportError.
        threshold: Optional minimum similarity threshold. Scores below this
            are clamped to 0.0 (useful for treating low similarity as failure).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_tfidf_fallback: bool = True,
        threshold: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._use_tfidf_fallback = use_tfidf_fallback
        self._threshold = threshold
        self._sentence_model: Any = None
        self._tried_import = False

    @property
    def name(self) -> str:
        return "similarity"

    def _get_sentence_model(self) -> Any:
        if self._tried_import:
            return self._sentence_model
        self._tried_import = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._sentence_model = SentenceTransformer(self._model_name)
            logger.info("Loaded sentence-transformers model '%s'", self._model_name)
        except ImportError:
            if not self._use_tfidf_fallback:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install it with: pip install sentence-transformers"
                )
            logger.info(
                "sentence-transformers not available; using TF-IDF cosine similarity fallback"
            )
        return self._sentence_model

    def _dense_similarity(self, output: str, expected: str) -> float:
        model = self._sentence_model
        import numpy as np  # type: ignore

        vecs = model.encode([output, expected], convert_to_numpy=True, normalize_embeddings=True)
        return float(np.dot(vecs[0], vecs[1]))

    def _tfidf_similarity(self, output: str, expected: str) -> float:
        tokens_a = _tokenize(output)
        tokens_b = _tokenize(expected)
        # Simple IDF: treat each document's vocab as the corpus (2-doc IDF)
        vocab = set(tokens_a) | set(tokens_b)
        idf: dict[str, float] = {}
        for term in vocab:
            doc_count = (1 if term in tokens_a else 0) + (1 if term in tokens_b else 0)
            idf[term] = math.log((2.0 + 1) / (doc_count + 1)) + 1.0
        vec_a = _tfidf_vector(tokens_a, idf)
        vec_b = _tfidf_vector(tokens_b, idf)
        return _cosine_similarity(vec_a, vec_b)

    def score(
        self,
        output: str,
        expected: str | None = None,
        **kwargs: Any,
    ) -> Score:
        if expected is None:
            logger.warning("SemanticSimilarityScorer called without expected; returning 0.0")
            return Score(
                value=0.0,
                scorer=self.name,
                reasoning="No expected answer provided.",
            )

        model = self._get_sentence_model()

        if model is not None:
            similarity = self._dense_similarity(output, expected)
            method = "sentence-transformers"
        else:
            similarity = self._tfidf_similarity(output, expected)
            method = "tfidf-fallback"

        # Clamp to [0, 1] — dot product of normalized vectors is in [-1, 1]
        similarity = max(0.0, min(1.0, similarity))

        if similarity < self._threshold:
            value = 0.0
            reasoning = (
                f"Similarity {similarity:.3f} below threshold {self._threshold:.3f} ({method})."
            )
        else:
            value = similarity
            reasoning = f"Cosine similarity: {similarity:.3f} ({method})."

        logger.debug("SemanticSimilarityScorer: similarity=%.3f, method=%s", similarity, method)

        return Score(
            value=value,
            scorer=self.name,
            reasoning=reasoning,
            metadata={"similarity": similarity, "method": method, "threshold": self._threshold},
        )
