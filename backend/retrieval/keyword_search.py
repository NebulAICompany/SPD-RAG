"""
Simple Keyword Search System using BM25 Algorithm
Integrates with existing vector search pipeline for hybrid retrieval
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
    """Represents a keyword search result"""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    matched_terms: List[str]


def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenizer to convert text into searchable terms
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of lowercase tokens (words)
    """
    if not text:
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace punctuation with spaces, keep alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into tokens
    tokens = text.split()
    
    # Filter out very short tokens (1 char) and very long tokens (>50 chars, likely noise)
    tokens = [t for t in tokens if 2 <= len(t) <= 50]
    
    return tokens


class BM25KeywordSearch:
    """BM25-based keyword search"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b  # Length normalization parameter

        self.documents: Dict[str, Dict[str, Any]] = {}
        self.term_frequencies: Dict[str, Dict[str, int]] = {}
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.document_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.total_documents: int = 0

        # File paths for persistence
        self.index_dir = Path(VECTORSTORE_PATH_STR) / "keyword_index"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.documents_file = self.index_dir / "documents.json"
        self.index_file = self.index_dir / "bm25_index.pkl"

        logger.info("🔍 BM25 Keyword Search initialized")

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        terms: List[str] = None,
    ):
        """Add a document to the search index with pre-processed terms"""
        if not content or not content.strip():
            logger.warning(f"⚠️ Skipping document {doc_id}: empty content")
            return

        # Use provided terms or tokenize the content automatically
        if terms is None:
            terms = tokenize_text(content)
            if not terms:
                logger.warning(f"⚠️ No terms extracted from document {doc_id}")
                return
            logger.debug(f"🔤 Tokenized {len(terms)} terms from document {doc_id}")

        # Store document
        self.documents[doc_id] = {
            "content": content,
            "metadata": metadata,
            "terms": terms,
        }

        # Calculate term frequencies for this document
        term_freq = Counter(terms)
        self.term_frequencies[doc_id] = dict(term_freq)

        # Update document frequencies (how many docs contain each term)
        unique_terms = set(terms)
        for term in unique_terms:
            self.document_frequencies[term] += 1

        # Store document length
        self.document_lengths[doc_id] = len(terms)

        # Update total documents and average length
        self.total_documents = len(self.documents)
        if self.total_documents > 0:
            self.avg_doc_length = (
                sum(self.document_lengths.values()) / self.total_documents
            )

        logger.debug(
            f"✅ Added document {doc_id} with {len(terms)} terms (total: {self.total_documents})"
        )

    def calculate_bm25_score(
        self, query_terms: List[str], doc_id: str
    ) -> Tuple[float, List[str]]:
        """Calculate BM25 score for a document given query terms"""
        if doc_id not in self.term_frequencies:
            # logger.debug(f"Document {doc_id} not found in term frequencies")
            # logger.debug(f"Term frequencies: {self.term_frequencies}")
            return 0.0, []
        # logger.debug(f"Calculating BM25 score for document {doc_id}")
        doc_tf = self.term_frequencies[doc_id]
        doc_length = self.document_lengths[doc_id]
        # logger.debug(f"Document ID: {doc_id}")
        # logger.debug(f"Document TFs: {self.term_frequencies}")
        # logger.debug(f"Document lengths: {self.document_lengths}")
        matched_terms = []
        score = 0.0
        # logger.debug(f"Query terms: {query_terms}")
        for term in query_terms:
            if term in doc_tf:
                matched_terms.append(term)
                # logger.debug(f"Matched term: {term}")
                # Term frequency in document
                tf = doc_tf[term]
                # logger.debug(f"Term frequency: {tf}")
                # Document frequency (how many docs contain this term)
                df = self.document_frequencies.get(term, 0)
                # logger.debug(f"Document frequency: {df}")
                if df == 0:
                    continue

                # IDF calculation
                idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
                # logger.debug(f"IDF: {idf}")
                # BM25 formula
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
    ) -> List[SearchResult]:
        """Search documents using BM25 algorithm with pre-processed query terms"""
        if not query_terms:
            return []

        if self.total_documents == 0:
            logger.warning("No documents in keyword search index")
            return []

        logger.info(f"🔍 Keyword search for terms: {query_terms} (limit: {k})")
        logger.info(f"Searching through all {len(self.documents)} documents")

        # Calculate scores for all documents
        scores = []
        for doc_id in self.documents:
            score, matched_terms = self.calculate_bm25_score(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score, matched_terms))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Create search results
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

        logger.info(f"✅ Found {len(results)} matching documents")
        return results

    def save_index(self):
        """Save the search index to disk"""
        try:
            # Ensure directory exists before saving
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # Save documents as JSON
            with open(self.documents_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)

            # Save index data as pickle
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

            logger.info(
                f"💾 Keyword search index saved ({self.total_documents} documents)"
            )

        except Exception as e:
            logger.error(f"❌ Failed to save keyword search index: {e}")

    def load_index(self) -> bool:
        """Load the search index from disk"""
        try:
            if not self.documents_file.exists() or not self.index_file.exists():
                logger.info("No existing keyword search index found")
                return False

            # Load documents
            with open(self.documents_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)

            # Load index data
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

            logger.info(
                f"📚 Keyword search index loaded ({self.total_documents} documents)"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load keyword search index: {e}")
            return False

    def clear_index(self):
        """Clear the entire search index"""
        self.documents.clear()
        self.term_frequencies.clear()
        self.document_frequencies.clear()
        self.document_lengths.clear()
        self.avg_doc_length = 0.0
        self.total_documents = 0

        # Remove index files
        try:
            if self.documents_file.exists():
                self.documents_file.unlink()
            if self.index_file.exists():
                self.index_file.unlink()
            logger.info("🗑️ Keyword search index cleared")
        except Exception as e:
            logger.error(f"❌ Failed to clear index files: {e}")

    def remove_documents_by_file(self, file_name: str):
        """Remove all documents from a specific file"""
        docs_to_remove = []

        # Find all document IDs that start with the file name
        for doc_id in self.documents.keys():
            if doc_id.startswith(f"{file_name}_"):
                docs_to_remove.append(doc_id)

        if not docs_to_remove:
            logger.info(f"No existing documents found for file: {file_name}")
            return

        logger.info(
            f"🗑️ Removing {len(docs_to_remove)} existing documents for file: {file_name}"
        )

        # Remove documents and their associated data
        for doc_id in docs_to_remove:
            # Remove from documents
            if doc_id in self.documents:
                del self.documents[doc_id]

            # Remove from term frequencies and update document frequencies
            if doc_id in self.term_frequencies:
                # Decrease document frequencies for each unique term in this document
                unique_terms = set(self.term_frequencies[doc_id].keys())
                for term in unique_terms:
                    if term in self.document_frequencies:
                        self.document_frequencies[term] -= 1
                        if self.document_frequencies[term] <= 0:
                            del self.document_frequencies[term]

                del self.term_frequencies[doc_id]

            # Remove from document lengths
            if doc_id in self.document_lengths:
                del self.document_lengths[doc_id]

        # Update total documents and average length
        self.total_documents = len(self.documents)
        if self.total_documents > 0:
            self.avg_doc_length = (
                sum(self.document_lengths.values()) / self.total_documents
            )
        else:
            self.avg_doc_length = 0.0

        logger.info(
            f"✅ Removed {len(docs_to_remove)} documents. Index now has {self.total_documents} documents"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get search index statistics"""
        return {
            "total_documents": self.total_documents,
            "total_terms": len(self.document_frequencies),
            "avg_doc_length": self.avg_doc_length,
            "index_size_mb": self._get_index_size_mb(),
        }

    def _get_index_size_mb(self) -> float:
        """Get index size in MB"""
        try:
            total_size = 0
            if self.documents_file.exists():
                total_size += self.documents_file.stat().st_size
            if self.index_file.exists():
                total_size += self.index_file.stat().st_size
            return total_size / (1024 * 1024)
        except:
            return 0.0


# Global keyword search instance
_keyword_search_instance = None


def get_keyword_search() -> BM25KeywordSearch:
    """Get or create the global keyword search instance"""
    global _keyword_search_instance
    if _keyword_search_instance is None:
        _keyword_search_instance = BM25KeywordSearch()
        _keyword_search_instance.load_index()
    return _keyword_search_instance


def keyword_search(
    query_terms: List[str], k: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform keyword search and return results in the same format as vector search

    Args:
        query_terms: Pre-processed search terms
        k: Number of results to return

    Returns:
        List of search results compatible with existing retrieval system
    """
    search_engine = get_keyword_search()
    results = search_engine.search(query_terms, k=k)

    # Convert to format compatible with existing retrieval system
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
