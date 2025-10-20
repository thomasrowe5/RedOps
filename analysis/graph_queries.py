"""Run analytic Cypher queries against a Neo4j attack graph."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TechniqueDegree:
    """Representation of a technique degree summary."""

    technique: str
    degree: int


@dataclass(frozen=True)
class TechniqueTransition:
    """Representation of a transition from one technique to another."""

    source: str
    target: str
    count: int


def _parse_timestamp(raw: Optional[Any]) -> Optional[datetime]:
    """Attempt to parse a timestamp string into a :class:`datetime`.

    Neo4j stores timestamps from the orchestrator as ISO-8601 strings. This helper
    mirrors the tolerant parsing behaviour used throughout the repository. When the
    timestamp cannot be parsed we return ``None`` and allow the caller to decide on
    an appropriate fallback ordering.
    """

    if raw in (None, ""):
        return None

    text = str(raw)
    # Normalise timestamps ending with ``Z`` into an explicit UTC offset so that
    # :func:`datetime.fromisoformat` can handle them.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _event_sort_key(event: Dict[str, Any]) -> Tuple[Any, Any, int]:
    """Sort events by timestamp, falling back to event-id ordering."""

    parsed = _parse_timestamp(event.get("timestamp"))
    if parsed is None:
        # Place unparseable timestamps at the end whilst keeping their relative
        # order via the original index within the relationship array.
        timestamp_key: Any = (1, event.get("sequence", 0))
    else:
        timestamp_key = (0, parsed)

    event_id = event.get("event_id")
    if isinstance(event_id, (int, float)):
        event_key: Any = (0, event_id)
    else:
        # Attempt to coerce common numeric string identifiers.
        try:
            event_key = (0, int(str(event_id)))
        except (TypeError, ValueError):
            event_key = (1, str(event_id) if event_id is not None else "")

    return timestamp_key, event_key, int(event.get("sequence", 0))


def top_techniques_by_degree(driver, limit: int = 20) -> List[TechniqueDegree]:
    """Return the techniques with the highest degree in the graph."""

    if limit <= 0:
        return []

    query = (
        "MATCH (t:Technique) "
        "RETURN t.name AS technique, size((t)--()) AS degree "
        "ORDER BY degree DESC, technique ASC "
        "LIMIT $limit"
    )

    def _tx_run(tx, limit_value: int) -> List[TechniqueDegree]:
        records = tx.run(query, limit=limit_value)
        return [TechniqueDegree(record["technique"], record["degree"]) for record in records]

    with driver.session() as session:
        results = session.read_transaction(_tx_run, limit)
        LOGGER.debug("Fetched %d technique degree results", len(results))
        return results


def technique_transition_counts(driver) -> List[TechniqueTransition]:
    """Compute the ordered transition counts between techniques for each agent."""

    cypher = (
        "MATCH (a:Agent)-[r:PERFORMS]->(t:Technique) "
        "RETURN a.name AS agent, t.name AS technique, "
        "       r.timestamps AS timestamps, r.event_ids AS event_ids"
    )

    def _tx_fetch(tx) -> Sequence[Dict[str, Any]]:
        return list(tx.run(cypher))

    with driver.session() as session:
        rows = session.read_transaction(_tx_fetch)

    events_by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        agent = row.get("agent") or "unknown-agent"
        technique = row.get("technique") or "unknown-technique"
        timestamps: Sequence[Any] = row.get("timestamps") or []
        event_ids: Sequence[Any] = row.get("event_ids") or []
        max_len = max(len(timestamps), len(event_ids))
        if max_len == 0:
            # No concrete event information for this relationship; treat it as a
            # single undated occurrence so that techniques with otherwise missing
            # metadata still contribute to degree calculations.
            events_by_agent[agent].append(
                {
                    "technique": technique,
                    "timestamp": None,
                    "event_id": None,
                    "sequence": 0,
                }
            )
            continue

        for index in range(max_len):
            events_by_agent[agent].append(
                {
                    "technique": technique,
                    "timestamp": timestamps[index] if index < len(timestamps) else None,
                    "event_id": event_ids[index] if index < len(event_ids) else None,
                    "sequence": index,
                }
            )

    transition_counter: Counter[Tuple[str, str]] = Counter()

    for agent, events in events_by_agent.items():
        if not events:
            continue
        sorted_events = sorted(events, key=_event_sort_key)
        for idx in range(len(sorted_events) - 1):
            current = sorted_events[idx]
            nxt = sorted_events[idx + 1]
            source = current.get("technique")
            target = nxt.get("technique")
            if not source or not target:
                continue
            transition_counter[(source, target)] += 1
        LOGGER.debug("Processed %d events for agent %s", len(events), agent)

    transitions = [
        TechniqueTransition(source=src, target=dst, count=count)
        for (src, dst), count in transition_counter.most_common()
    ]
    LOGGER.debug("Computed %d transition pairs", len(transitions))
    return transitions


def communities_louvain(driver) -> Dict[str, Any]:
    """Run the APOC Louvain community detection on techniques.

    When the APOC plugin (or the specific Louvain procedure) is not available the
    function returns a dictionary describing that the operation could not be
    completed rather than raising an exception. This allows CLI consumers to handle
    the situation gracefully.
    """

    node_query = "MATCH (t:Technique) RETURN id(t) AS id"
    rel_query = (
        "MATCH (a:Agent)-[:PERFORMS]->(t1:Technique) "
        "MATCH (a)-[:PERFORMS]->(t2:Technique) "
        "WHERE id(t1) <> id(t2) "
        "RETURN id(t1) AS source, id(t2) AS target"
    )

    query = (
        "CALL apoc.algo.louvainWithConfig($nodeQuery, $relQuery, {graph:'cypher'}) "
        "YIELD nodeId, community "
        "WITH nodeId, community "
        "MATCH (t:Technique) WHERE id(t) = nodeId "
        "RETURN community AS community_id, collect(t.name) AS techniques "
        "ORDER BY community_id"
    )

    def _run(tx):
        return list(
            tx.run(
                query,
                nodeQuery=node_query,
                relQuery=rel_query,
            )
        )

    try:
        with driver.session() as session:
            records = session.read_transaction(_run)
    except Neo4jError as exc:  # pragma: no cover - depends on external Neo4j instance
        if exc.code == "Neo.ClientError.Procedure.ProcedureNotFound":
            LOGGER.warning("APOC Louvain procedure not available: %s", exc)
            return {
                "status": "not available",
                "reason": "APOC procedure apoc.algo.louvainWithConfig not found.",
            }
        raise

    communities = [
        {
            "community": record["community_id"],
            "techniques": sorted(record.get("techniques") or []),
        }
        for record in records
    ]
    LOGGER.debug("Retrieved %d communities", len(communities))
    return {"status": "ok", "communities": communities}


def _build_driver(uri: str, username: Optional[str], password: Optional[str]):
    if username and password:
        return GraphDatabase.driver(uri, auth=(username, password))
    return GraphDatabase.driver(uri)


def _serialize_dataclasses(items: Iterable[Any]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for item in items:
        if is_dataclass(item):
            serialized.append(asdict(item))
        elif isinstance(item, dict):
            serialized.append(dict(item))
        else:
            serialized.append(dict(item.__dict__))  # type: ignore[arg-type]
    return serialized


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis queries against Neo4j")
    parser.add_argument("--neo4j", default="bolt://localhost:7687", help="Neo4j bolt URI")
    parser.add_argument("--username", default=None, help="Neo4j username")
    parser.add_argument("--password", default=None, help="Neo4j password")
    parser.add_argument(
        "--op",
        required=True,
        choices={"top", "transitions", "communities"},
        help="Analysis operation to execute",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit for results (applicable to --op top)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    try:
        driver = _build_driver(args.neo4j, args.username, args.password)
    except Exception as exc:  # pragma: no cover - depends on external driver
        LOGGER.error("Failed to create Neo4j driver: %s", exc)
        return 1

    try:
        with driver:
            if args.op == "top":
                results = top_techniques_by_degree(driver, args.limit)
                payload: Any = _serialize_dataclasses(results)
            elif args.op == "transitions":
                results = technique_transition_counts(driver)
                payload = _serialize_dataclasses(results)
            else:
                payload = communities_louvain(driver)
    except Neo4jError as exc:  # pragma: no cover - depends on external Neo4j instance
        LOGGER.error("Neo4j query failed: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.exception("Unexpected error running analysis: %s", exc)
        return 1

    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
