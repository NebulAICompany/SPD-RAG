"""BM25 keyword search for hybrid retrieval (vector + keyword).

Integrates with the dense retrieval pipeline: keyword results can be merged with
vector results (e.g. in retrieve_with_keyword_helping) for better recall on
lexical matches.
"""

import json
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass

from backend.shared.logger import get_logger
from backend.shared.constants import VECTORSTORE_PATH_STR

logger = get_logger("KEYWORD_SEARCH")


@dataclass
class SearchResult:
    """Single result from BM25 keyword search (doc_id, content, score, metadata, matched_terms)."""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    matched_terms: List[str]


def tokenize_text(text: str) -> List[str]:
    """Normalize text into BM25-style tokens: lowercase, alphanumeric, length 2-50.

    Args:
        text: Raw input text.

    Returns:
        List of tokens used for indexing and querying.
    """
    if not text:
        return []

    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if 2 <= len(t) <= 50]
    return tokens


class BM25KeywordSearch:
    """BM25 index and search: k1 (term frequency saturation), b (length normalization)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.documents: Dict[str, Dict[str, Any]] = {}
        self.term_frequencies: Dict[str, Dict[str, int]] = {}
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.document_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.total_documents: int = 0

        self.index_dir = Path(VECTORSTORE_PATH_STR) / "keyword_index"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.documents_file = self.index_dir / "documents.json"
        self.index_file = self.index_dir / "bm25_index.pkl"

        logger.info("BM25 keyword search initialized")

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        terms: List[str] = None,
    ):
        """Add a document to the BM25 index; tokenizes content if terms not provided."""
        if not content or not content.strip():
            logger.warning("Skipping empty document: %s", doc_id)
            return

        if terms is None:
            terms = tokenize_text(content)
            if not terms:
                logger.warning("No terms extracted from document: %s", doc_id)
                return
            logger.debug("Tokenized %s terms from document %s", len(terms), doc_id)

        self.documents[doc_id] = {
            "content": content,
            "metadata": metadata,
            "terms": terms,
        }

        term_freq = Counter(terms)
        self.term_frequencies[doc_id] = dict(term_freq)

        unique_terms = set(terms)
        for term in unique_terms:
            self.document_frequencies[term] += 1

        self.document_lengths[doc_id] = len(terms)

        self.total_documents = len(self.documents)
        if self.total_documents > 0:
            self.avg_doc_length = (
                sum(self.document_lengths.values()) / self.total_documents
            )

        logger.debug("Added document %s (%s terms, total docs: %s)", doc_id, len(terms), self.total_documents)

    def calculate_bm25_score(
        self, query_terms: List[str], doc_id: str
    ) -> Tuple[float, List[str]]:
        """Compute BM25 score for one document and return matched query terms."""
        if doc_id not in self.term_frequencies:
            return 0.0, []

        doc_tf = self.term_frequencies[doc_id]
        doc_length = self.document_lengths[doc_id]
        matched_terms = []
        score = 0.0

        for term in query_terms:
            if term in doc_tf:
                matched_terms.append(term)
                tf = doc_tf[term]
                df = self.document_frequencies.get(term, 0)
                if df == 0:
                    continue

                idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )

                term_score = idf * (numerator / denominator)
                score += term_score

        return score, matched_terms

    def search(
        self,
        query_terms: List[str],
        k: int = 10,
        selected_files: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Return top-k documents by BM25 score; optionally filter by selected_files."""
        if not query_terms:
            return []

        if self.total_documents == 0:
            logger.warning("Keyword index is empty")
            return []

        logger.info("Keyword search terms=%s limit=%s", query_terms, k)

        logger.info(f"Selected files: {selected_files}")
        logger.info(f"Length of Documents: {len(self.documents)}")

        scores = []
        for doc_id in self.documents:
            if selected_files:
                doc_metadata = self.documents[doc_id]["metadata"]
                file_name = doc_metadata.get("file_name", "")
                if file_name not in [selected_file.split(".")[0] if "." in selected_file else selected_file for selected_file in selected_files]:
                    continue

            score, matched_terms = self.calculate_bm25_score(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score, matched_terms))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score, matched_terms in scores[:k]:
            doc_data = self.documents[doc_id]
            result = SearchResult(
                doc_id=doc_id,
                content=doc_data["content"],
                score=score,
                metadata=doc_data["metadata"],
                matched_terms=matched_terms,
            )
            results.append(result)

        logger.info("Keyword search returned %s results", len(results))
        return results

    def save_index(self):
        """Persist the BM25 index (documents JSON + term stats pickle) to disk."""
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)

            with open(self.documents_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)

            index_data = {
                "term_frequencies": self.term_frequencies,
                "document_frequencies": dict(self.document_frequencies),
                "document_lengths": self.document_lengths,
                "avg_doc_length": self.avg_doc_length,
                "total_documents": self.total_documents,
                "k1": self.k1,
                "b": self.b,
            }

            with open(self.index_file, "wb") as f:
                pickle.dump(index_data, f)

            logger.info("Keyword index saved (%s documents)", self.total_documents)

        except Exception as e:
            logger.error("Failed to save keyword index: %s", e)

    def load_index(self) -> bool:
        """Load the BM25 index from disk; returns False if no index exists."""
        try:
            if not self.documents_file.exists() or not self.index_file.exists():
                logger.info("No existing keyword search index found")
                return False

            with open(self.documents_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)

            with open(self.index_file, "rb") as f:
                index_data = pickle.load(f)

            self.term_frequencies = index_data["term_frequencies"]
            self.document_frequencies = defaultdict(
                int, index_data["document_frequencies"]
            )
            self.document_lengths = index_data["document_lengths"]
            self.avg_doc_length = index_data["avg_doc_length"]
            self.total_documents = index_data["total_documents"]
            self.k1 = index_data.get("k1", 1.5)
            self.b = index_data.get("b", 0.75)

            logger.info("Keyword index loaded (%s documents)", self.total_documents)
            return True

        except Exception as e:
            logger.error("Failed to load keyword index: %s", e)
            return False

    def clear_index(self):
        """Clear the in-memory index and delete persisted index files."""
        self.documents.clear()
        self.term_frequencies.clear()
        self.document_frequencies.clear()
        self.document_lengths.clear()
        self.avg_doc_length = 0.0
        self.total_documents = 0

        try:
            if self.documents_file.exists():
                self.documents_file.unlink()
            if self.index_file.exists():
                self.index_file.unlink()
            logger.info("Keyword index cleared")
        except Exception as e:
            logger.error("Failed to clear index files: %s", e)

    def remove_documents_by_file(self, file_name: str):
        """Remove all indexed chunks belonging to the given file name."""
        docs_to_remove = []

        for doc_id in self.documents.keys():
            if doc_id.startswith(f"{file_name}_"):
                docs_to_remove.append(doc_id)

        if not docs_to_remove:
            logger.info("No documents to remove for file: %s", file_name)
            return

        logger.info("Removing %s documents for file: %s", len(docs_to_remove), file_name)

        for doc_id in docs_to_remove:
            if doc_id in self.documents:
                del self.documents[doc_id]

            if doc_id in self.term_frequencies:
                unique_terms = set(self.term_frequencies[doc_id].keys())
                for term in unique_terms:
                    if term in self.document_frequencies:
                        self.document_frequencies[term] -= 1
                        if self.document_frequencies[term] <= 0:
                            del self.document_frequencies[term]

                del self.term_frequencies[doc_id]

            if doc_id in self.document_lengths:
                del self.document_lengths[doc_id]

        self.total_documents = len(self.documents)
        if self.total_documents > 0:
            self.avg_doc_length = (
                sum(self.document_lengths.values()) / self.total_documents
            )
        else:
            self.avg_doc_length = 0.0

        logger.info("Removed %s documents; index has %s documents", len(docs_to_remove), self.total_documents)

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics: total_documents, total_terms, avg_doc_length, index_size_mb."""
        return {
            "total_documents": self.total_documents,
            "total_terms": len(self.document_frequencies),
            "avg_doc_length": self.avg_doc_length,
            "index_size_mb": self._get_index_size_mb(),
        }

    def _get_index_size_mb(self) -> float:
        """Return total size of persisted index files in megabytes."""
        try:
            total_size = 0
            if self.documents_file.exists():
                total_size += self.documents_file.stat().st_size
            if self.index_file.exists():
                total_size += self.index_file.stat().st_size
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0


_keyword_search_instance = None


def get_keyword_search() -> BM25KeywordSearch:
    """Return the global BM25 instance, loading the index from disk if needed."""
    global _keyword_search_instance
    if _keyword_search_instance is None:
        _keyword_search_instance = BM25KeywordSearch()
        _keyword_search_instance.load_index()
    return _keyword_search_instance


def keyword_search(
    query_terms: List[str], k: int = 10, selected_files: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Run BM25 search and return results in the same shape as vector retrieval.

    Args:
        query_terms: Tokenized query terms (e.g. from tokenize_text).
        k: Maximum number of results.
        selected_files: Optional list of file names to restrict the search.

    Returns:
        List of dicts with content, score, metadata (match_type='keyword_match', etc.).
    """
    search_engine = get_keyword_search()
    results = search_engine.search(query_terms, k=k, selected_files=selected_files)

    formatted_results = []
    for result in results:
        formatted_result = {
            "content": result.content,
            "score": result.score,
            "metadata": {
                **result.metadata,
                "match_type": "keyword_match",
                "matched_terms": result.matched_terms,
                "search_method": "bm25",
            },
        }
        formatted_results.append(formatted_result)

    return formatted_results
