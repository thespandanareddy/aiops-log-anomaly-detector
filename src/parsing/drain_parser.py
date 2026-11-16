"""
Drain3 Log Parser Wrapper with Persistent State.
Performs streaming prefix-tree log template extraction and masking.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

from src.config import (
    DRAIN_DEPTH,
    DRAIN_MAX_CHILDREN,
    DRAIN_PARAM_STR,
    DRAIN_SIM_TH,
    DRAIN_STATE_FILE,
)
from src.preprocessing.maskers import masker

logger = logging.getLogger(__name__)


class DrainParser:
    """Wrapper for Drain3 streaming log template miner."""

    def __init__(
        self,
        persistence_path: Optional[Path] = DRAIN_STATE_FILE,
        sim_th: float = DRAIN_SIM_TH,
        depth: int = DRAIN_DEPTH,
        max_children: int = DRAIN_MAX_CHILDREN,
    ):
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.config = TemplateMinerConfig()
        self.config.drain_sim_th = sim_th
        self.config.drain_depth = depth
        self.config.drain_max_children = max_children
        self.config.parametrize_numeric_tokens = True

        self.persistence = None
        if self.persistence_path:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self.persistence = FilePersistence(str(self.persistence_path))

        self.miner = TemplateMiner(
            persistence_handler=self.persistence,
            config=self.config
        )
        self.template_cache: Dict[str, str] = {}
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        """Sync internal dictionary of event_id -> template string."""
        self.template_cache.clear()
        if hasattr(self.miner, "drain") and hasattr(self.miner.drain, "id_to_cluster"):
            for cluster_id, cluster in self.miner.drain.id_to_cluster.items():
                event_id = f"E{cluster_id}"
                self.template_cache[event_id] = cluster.get_template()

    def parse_message(
        self,
        content: str,
        update_miner: bool = True
    ) -> Tuple[str, str, bool]:
        """
        Parse raw log content using masking + Drain3.
        Returns:
            (event_id, template_str, is_new_cluster)
        """
        masked = masker.mask(content)
        if not masked.strip():
            return "E0", "<EMPTY_LOG>", False

        if update_miner:
            result = self.miner.add_log_message(masked)
            cluster_id = result["cluster_id"]
            template_str = result["template_mined"]
            is_new = result["change_type"] == "cluster_created"
            event_id = f"E{cluster_id}"
            self.template_cache[event_id] = template_str
            return event_id, template_str, is_new
        else:
            # Read-only match without altering cluster state
            match = self.miner.match(masked)
            if match is not None:
                event_id = f"E{match.cluster_id}"
                template_str = match.get_template()
                return event_id, template_str, False
            return "E_UNSEEN", masked, True

    def get_template(self, event_id: str) -> Optional[str]:
        """Lookup template text by event ID."""
        return self.template_cache.get(event_id)

    def get_all_templates(self) -> Dict[str, str]:
        """Return snapshot of all extracted templates."""
        self._refresh_cache()
        return dict(self.template_cache)

    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Persist miner state to disk."""
        if self.persistence:
            self.miner.save_state("manual_save")
        elif filepath:
            fp = FilePersistence(str(filepath))
            fp.save_state(self.miner.drain.get_state())

        # Also save a human-readable JSON map
        json_path = (filepath or self.persistence_path).with_suffix(".json") if (filepath or self.persistence_path) else None
        if json_path:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.get_all_templates(), f, indent=2)

    def load_state(self, filepath: Optional[Path] = None) -> None:
        """Reload state from disk."""
        target = filepath or self.persistence_path
        if target and target.exists():
            fp = FilePersistence(str(target))
            self.miner = TemplateMiner(persistence_handler=fp, config=self.config)
            self._refresh_cache()
            logger.info(f"Loaded {len(self.template_cache)} templates from {target}")


# Global singleton instance
parser = DrainParser()
