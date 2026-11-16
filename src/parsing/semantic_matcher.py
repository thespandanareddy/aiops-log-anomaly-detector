"""
Semantic and Exact Event Matcher.
Enables tolerance to slight wording variations and minor token edits
by combining exact Drain3 template matching with TF-IDF cosine similarity fallback.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import SEMANTIC_SIM_THRESHOLD
from src.parsing.drain_parser import DrainParser
from src.preprocessing.maskers import masker

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """Matches raw or masked log messages against known templates using exact and semantic logic."""

    def __init__(
        self,
        drain_parser: DrainParser,
        similarity_threshold: float = SEMANTIC_SIM_THRESHOLD
    ):
        self.drain_parser = drain_parser
        self.similarity_threshold = similarity_threshold
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.template_ids: List[str] = []
        self.template_texts: List[str] = []
        self.template_vectors: Optional[np.ndarray] = None
        self.rebuild_index()

    def rebuild_index(self) -> None:
        """Fit or update the TF-IDF representation over all known Drain3 templates."""
        templates_map = self.drain_parser.get_all_templates()
        if not templates_map:
            self.vectorizer = None
            self.template_ids = []
            self.template_texts = []
            self.template_vectors = None
            return

        self.template_ids = list(templates_map.keys())
        self.template_texts = [templates_map[tid] for tid in self.template_ids]

        # Use character + word n-grams for typo & slight phrasing tolerance
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True
        )
        self.template_vectors = self.vectorizer.fit_transform(self.template_texts)

    def match(
        self,
        raw_message: str,
        allow_semantic_fallback: bool = True
    ) -> Dict[str, object]:
        """
        Match a log message to an event template.
        Returns a dict:
            {
                "event_id": str,
                "template": str,
                "confidence": float,
                "match_type": "EXACT" | "SEMANTIC" | "NOVEL",
                "is_novel": bool,
                "raw_message": str,
                "masked_message": str
            }
        """
        masked = masker.mask(raw_message)

        # 1. Try exact Drain3 match
        event_id, template_str, is_new = self.drain_parser.parse_message(
            raw_message, update_miner=False
        )
        if event_id != "E_UNSEEN" and not is_new:
            return {
                "event_id": event_id,
                "template": template_str,
                "confidence": 1.0,
                "match_type": "EXACT",
                "is_novel": False,
                "raw_message": raw_message,
                "masked_message": masked,
            }

        # 2. If exact match fails, attempt semantic fallback
        if allow_semantic_fallback and self.vectorizer is not None and self.template_vectors is not None:
            if not self.template_texts:
                self.rebuild_index()

            if self.vectorizer and self.template_vectors is not None and len(self.template_ids) > 0:
                query_vec = self.vectorizer.transform([masked])
                sims = cosine_similarity(query_vec, self.template_vectors)[0]
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])

                if best_sim >= self.similarity_threshold:
                    matched_id = self.template_ids[best_idx]
                    matched_template = self.template_texts[best_idx]
                    return {
                        "event_id": matched_id,
                        "template": matched_template,
                        "confidence": round(best_sim, 4),
                        "match_type": "SEMANTIC",
                        "is_novel": False,
                        "raw_message": raw_message,
                        "masked_message": masked,
                    }

        # 3. Message does not match known patterns -> Flag as NOVEL / unseen
        return {
            "event_id": "E_UNSEEN",
            "template": masked,
            "confidence": 0.0,
            "match_type": "NOVEL",
            "is_novel": True,
            "raw_message": raw_message,
            "masked_message": masked,
        }
