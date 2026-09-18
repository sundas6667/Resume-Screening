"""
core/vector_store.py
=====================
Lightweight TF-IDF vector storage and semantic similarity  built entirely
on scikit-learn per the project's constraints (no FAISS/Chroma/Pinecone/
Milvus, no embedding models, no GPU).

Workflow: fit one shared vocabulary across the job description + every
candidate's resume text in a screening session, then compare via cosine
similarity. Sharing one vocabulary (rather than vectorizing each document
independently) is what makes the similarity scores comparable across
candidates , this is the "vectorization & semantic matching" step from the
architecture diagram, feeding into core/matcher.py and core/ranking.py.

The fitted vectorizer and document matrix are cached on the instance after
build(), so repeated queries (e.g. the chatbot asking "who else looks like
a good fit for X") reuse the existing vectors instead of recomputing them —
this is the module's whole reason for existing as a stateful "store" rather
than a stateless function.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import SIMILARITY_ENGINE_VERSION, PerformanceConfig
from utils.helpers import get_logger, timed

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when the store can't be built or queried  e.g. every document
    turned out to be empty or pure stopwords, leaving an empty vocabulary.
    Always caught by the caller (core/matcher.py's match_candidates), which
    proceeds without a semantic similarity score rather than failing the batch.
    """


class VectorStore:
    """Session scoped TF-IDF store for one job description + its batch of
    candidate resumes. Not a persistent/multi-session database  per the
    project's scope, a new screening run (new JD or new candidate batch)
    means a fresh `build()` call, not an update to an existing index.
    """

    def __init__(self, performance_config: Optional[PerformanceConfig] = None):
        self.performance_config = performance_config or PerformanceConfig()
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None  # sparse matrix: row 0 = JD, rows 1..N = candidates
        self._candidate_ids: List[str] = []
        self._cached_scores: Optional[Dict[str, float]] = None
        self.built_at: Optional[datetime] = None
        self._document_word_counts: List[int] = []

    @property
    def is_built(self) -> bool:
        return self._matrix is not None

    @timed(logger)
    def build(self, jd_text: str, candidate_texts: Dict[str, str]) -> None:
        """Fit a shared TF-IDF vocabulary across the JD and all candidates.

        Args:
            jd_text: cleaned job description text.
            candidate_texts: {candidate_id: resume_text}, in the order
                similarity results will be reported.

        Raises:
            VectorStoreError: if there's nothing usable to vectorize (e.g.
                every document is empty or entirely stopwords).
        """
        if not candidate_texts:
            raise VectorStoreError("Cannot build a vector store with zero candidates.")
        if not jd_text or not jd_text.strip():
            raise VectorStoreError("Cannot build a vector store with an empty job description.")

        self._candidate_ids = list(candidate_texts.keys())
        documents = [jd_text] + [candidate_texts[cid] or "" for cid in self._candidate_ids]

        self._vectorizer = TfidfVectorizer(
            max_features=self.performance_config.tfidf_max_features,
            stop_words="english",
            ngram_range=(1, 2),  # unigrams + bigrams, so phrases like "machine learning" count too
            sublinear_tf=True,   # dampens the effect of very high raw term counts
            lowercase=True,
        )
        try:
            self._matrix = self._vectorizer.fit_transform(documents)
        except ValueError as exc:
            # scikit-learn raises this when the resulting vocabulary is empty
            # (e.g. every document was blank or pure stopwords/punctuation).
            self._vectorizer = None
            self._matrix = None
            raise VectorStoreError(
                "Could not build a semantic vector space — the job description and/or resumes "
                "may be too short or contain no meaningful text."
            ) from exc

        self._cached_scores = None
        self.built_at = datetime.now()
        self._document_word_counts = [len(doc.split()) for doc in documents]
        logger.info(
            "Vector store built: %d candidates, vocabulary size %d",
            len(self._candidate_ids), len(self._vectorizer.vocabulary_),
        )

    def similarity_scores(self) -> Dict[str, float]:
        """Cosine similarity of every stored candidate against the JD,
        scaled to 0-100. Cached after first call , call build() again
        (with a new JD or candidate set) to invalidate.
        """
        self._ensure_built()
        if self._cached_scores is not None:
            return self._cached_scores

        jd_vector = self._matrix[0:1]
        candidate_matrix = self._matrix[1:]
        sims = cosine_similarity(jd_vector, candidate_matrix)[0]
        self._cached_scores = {
            cid: round(float(score) * 100, 1) for cid, score in zip(self._candidate_ids, sims)
        }
        return self._cached_scores

    def get_similarity(self, candidate_id: str) -> Optional[float]:
        """Similarity score for one candidate, or None if not in the store."""
        return self.similarity_scores().get(candidate_id)

    def similarity_search(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Rank stored candidates against an arbitrary new query text (e.g.
        a chatbot question like "who has cloud infrastructure experience"),
        reusing the already-fitted vocabulary rather than refitting.

        Returns [(candidate_id, score_0_to_100), ...] sorted descending,
        truncated to top_k.
        """
        self._ensure_built()
        if not query_text or not query_text.strip():
            return []
        query_vector = self._vectorizer.transform([query_text])
        candidate_matrix = self._matrix[1:]
        sims = cosine_similarity(query_vector, candidate_matrix)[0]
        ranked = sorted(zip(self._candidate_ids, sims), key=lambda pair: pair[1], reverse=True)
        return [(cid, round(float(score) * 100, 1)) for cid, score in ranked[:top_k]]

    def vocabulary_size(self) -> int:
        self._ensure_built()
        return len(self._vectorizer.vocabulary_)

    def get_candidate_vector(self, candidate_id: str):
        """Raw TF-IDF sparse vector row for one candidate, or None if not
        in the store. Exposed for debugging and future features (e.g. a
        future embedding-based similarity_search variant) rather than any
        current consumer.
        """
        self._ensure_built()
        if candidate_id not in self._candidate_ids:
            return None
        row_index = self._candidate_ids.index(candidate_id) + 1  # +1: row 0 is the JD
        return self._matrix[row_index]

    def get_job_vector(self):
        """Raw TF-IDF sparse vector row for the job description."""
        self._ensure_built()
        return self._matrix[0:1]

    def clear_cache(self) -> None:
        """Drop the memoized similarity_scores() result without discarding
        the fitted vectorizer/matrix , similarity_search() and
        get_similarity() still work after this; only the next
        similarity_scores() call recomputes. Mainly useful for tests and a
        future "reset" affordance in Settings.
        """
        self._cached_scores = None

    @property
    def metadata(self) -> Dict[str, object]:
        """Lightweight diagnostics snapshot  not persisted (this store is
        session-scoped and rebuilt per JD/batch), just useful for a debug
        panel or log line.
        """
        avg_words = (
            round(sum(self._document_word_counts) / len(self._document_word_counts), 1)
            if self._document_word_counts else 0.0
        )
        return {
            "candidate_count": len(self._candidate_ids),
            "vocabulary_size": self.vocabulary_size() if self.is_built else 0,
            "average_document_length_words": avg_words,
            "engine_version": SIMILARITY_ENGINE_VERSION,
            "built_at": self.built_at.isoformat() if self.built_at else None,
            "is_built": self.is_built,
        }

    def _ensure_built(self) -> None:
        if not self.is_built:
            raise VectorStoreError("VectorStore.build() must be called before querying similarity.")
