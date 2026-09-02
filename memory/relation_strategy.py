"""Relation-label strategies for the MAGMA relation-schema experiment.

The strategy only enriches edges selected by the existing MAGMA builders.  It
never creates, removes, redirects, or changes the LinkType/sub_type of an edge;
those fields remain available to the existing retriever.
"""

import json
import hashlib
import logging
import re
from collections import Counter, defaultdict
from typing import Dict

from .graph_db import LinkType


logger = logging.getLogger(__name__)

# These links define graph structure or entity/session membership and therefore
# must be identical in all three experiment conditions.
STRUCTURAL_SUBTYPES = {
    "SAME_ENTITY",
    "CONTAINS",
    "PART_OF",
    "BELONGS_TO_SESSION",
    "REFERS_TO",
    "MENTIONED_IN",
}


def _node_text(node) -> str:
    if node is None:
        return ""
    for field in ("content_narrative", "summary", "title"):
        value = getattr(node, field, "")
        if value:
            return str(value)[:1200]
    return str(node)[:1200]


def _safe_snake_case(value: str, fallback: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip()).strip("_").lower()
    if not value or value in {"related_to", "associated_with", "connected_to"}:
        return fallback
    return value[:120]


class RelationStrategy:
    """Base strategy. Original mode intentionally performs no mutations."""

    mode = "original"

    def __init__(self, llm_controller=None):
        self.llm_controller = llm_controller

    def apply(self, graph_db) -> int:
        return 0


class _GeneratedRelationStrategy(RelationStrategy):
    response_key = "sub_relation"
    batch_size = 100
    include_category = True

    def apply(self, graph_db) -> int:
        edge_groups = defaultdict(list)
        for link in graph_db.links.values():
            original_subtype = str(link.properties.get("sub_type", "relation")).lower()
            if original_subtype.upper() in STRUCTURAL_SUBTYPES:
                continue
            if link.link_type not in {LinkType.SEMANTIC, LinkType.TEMPORAL, LinkType.CAUSAL}:
                continue
            source = graph_db.get_node(link.source_node_id)
            target = graph_db.get_node(link.target_node_id)
            key = (
                link.source_node_id,
                link.target_node_id,
                link.link_type.value,
                original_subtype,
                _node_text(source)[:350],
                _node_text(target)[:350],
            )
            edge_groups[key].append(link)

        keys = list(edge_groups)
        generated_by_key = {}
        for start in range(0, len(keys), self.batch_size):
            batch = keys[start:start + self.batch_size]
            generated_by_key.update(self._generate_batch(batch))

        enriched = 0
        for key, links in edge_groups.items():
            original_subtype = key[3]
            generated = generated_by_key.get(key, {})
            relation = _safe_snake_case(
                generated.get(self.response_key, generated.get("relation", "")),
                f"{original_subtype}_relation",
            )
            for link in links:
                # Preserve sub_type because MAGMA uses it for directionality and
                # special traversal rules. sub_relation is the experimental label.
                link.properties.update({
                    "relation_mode": self.mode,
                    "sub_relation": relation,
                    "relation_description": str(generated.get("description", ""))[:500],
                    "relation_confidence": self._confidence(generated.get("confidence", 0.0)),
                })
                if self.mode == "free":
                    link.properties["free_relation"] = relation
                enriched += 1
        return enriched

    @staticmethod
    def _confidence(value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _generate_batch(self, keys) -> Dict:
        fallback = {
            key: {
                self.response_key: f"{key[3]}_relation",
                "description": "generation fallback",
                "confidence": 0.0,
            }
            for key in keys
        }
        if not self.llm_controller or not hasattr(self.llm_controller, "llm"):
            return fallback
        records = []
        for index, key in enumerate(keys):
            record = {"index": index, "source": key[4], "target": key[5]}
            if self.include_category:
                record["category"] = key[2]
            records.append(record)
        try:
            response = self.llm_controller.llm.get_completion(
                self.batch_prompt(records),
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=8000,
            )
            parsed = json.loads(response).get("relations", [])
            for item in parsed:
                index = int(item.get("index", -1))
                if 0 <= index < len(keys):
                    fallback[keys[index]] = item
            return fallback
        except Exception as exc:
            logger.warning("Relation generation failed; using deterministic fallback: %s", exc)
            return fallback

    def batch_prompt(self, records) -> str:
        raise NotImplementedError


class FreeRelationStrategy(_GeneratedRelationStrategy):
    mode = "free"
    response_key = "relation"
    include_category = False

    def batch_prompt(self, records) -> str:
        return f"""You are extracting precise relationships between pairs of memory events.

For every indexed source-target pair, generate a short, specific snake_case
relation describing how the source relates to the target. Do not choose from a
predefined relation vocabulary. Avoid generic names such as related_to,
associated_with, and connected_to.

Examples:
"Trip schedule changed" -> "Hotel reservation was changed":
schedule_change_triggered_reservation_change
"Started learning Python" -> "Used PyTorch in research":
learning_progressed_to_research_use
"Discussed Kyoto travel" -> "Discussed Osaka travel": similar_travel_topic

Pairs:
{json.dumps(records, ensure_ascii=False)}

Return every input index exactly once as JSON only:
{{"relations": [{{"index": 0, "relation": "...", "description": "...", "confidence": 0.0}}]}}"""


class HybridRelationStrategy(_GeneratedRelationStrategy):
    mode = "hybrid"
    response_key = "sub_relation"

    def batch_prompt(self, records) -> str:
        return f"""You are extracting hierarchical relationships between pairs of memory events.

For every indexed pair, its upper-level relation category has already been
determined as SEMANTIC, TEMPORAL, or CAUSAL. Generate a short and precise
snake_case sub-relation describing the specific meaning. Do not change the
upper-level category.

Examples:
TEMPORAL / "Planned Kyoto trip" -> "Booked hotel": planning_preceded_booking
CAUSAL / "Travel schedule changed" -> "Hotel reservation changed":
reservation_changed_due_to_schedule
SEMANTIC / "Discussed Kyoto hotel" -> "Discussed Osaka hotel":
same_travel_accommodation_topic

Pairs:
{json.dumps(records, ensure_ascii=False)}

Return every input index exactly once as JSON only:
{{"relations": [{{"index": 0, "sub_relation": "...", "description": "...", "confidence": 0.0}}]}}"""


def create_relation_strategy(mode: str, llm_controller=None) -> RelationStrategy:
    strategies = {
        "original": RelationStrategy,
        "free": FreeRelationStrategy,
        "hybrid": HybridRelationStrategy,
    }
    try:
        return strategies[mode](llm_controller)
    except KeyError as exc:
        raise ValueError(f"Unknown relation mode: {mode}") from exc


def relation_statistics(graph_db, mode: str) -> Dict:
    """Return serializable edge-label statistics for one run."""
    type_counts = Counter()
    subtype_counts = Counter()
    generated_counts = Counter()
    generated_by_type = defaultdict(set)
    topology_rows = []

    for link in graph_db.links.values():
        link_type = link.link_type.value
        subtype = str(link.properties.get("sub_type", "UNKNOWN"))
        type_counts[link_type] += 1
        subtype_counts[subtype] += 1
        relation = link.properties.get("sub_relation")
        topology_rows.append((link.source_node_id, link.target_node_id, link_type, subtype))
        if relation:
            generated_counts[relation] += 1
            generated_by_type[link_type].add(relation)

    result = {
        "relation_mode": mode,
        "link_type_counts": dict(sorted(type_counts.items())),
        "link_subtype_counts": dict(sorted(subtype_counts.items())),
        "topology_sha256": hashlib.sha256(
            json.dumps(sorted(topology_rows), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    if mode in {"free", "hybrid"}:
        result.update({
            "unique_relations": len(generated_counts),
            "single_use_relations": sum(count == 1 for count in generated_counts.values()),
            "relation_counts": dict(sorted(generated_counts.items())),
        })
    if mode == "hybrid":
        result["unique_sub_relations_by_link_type"] = {
            key: len(values) for key, values in sorted(generated_by_type.items())
        }
    return result
