"""Utilities for building attack graphs in Neo4j from orchestrator events."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from neo4j import GraphDatabase, basic_auth


LOGGER = logging.getLogger(__name__)
DEFAULT_NEO4J_URI = "bolt://neo4j:7687"
DEFAULT_EVENTS_DIR = Path("./orchestrator/data/runs")

# Module level configuration updated by CLI helpers
_NEO4J_URI: str = DEFAULT_NEO4J_URI
_NEO4J_AUTH: Optional[Tuple[str, str]] = None


def configure_neo4j(uri: str, auth: Optional[Tuple[str, str]] = None) -> None:
    """Configure the target Neo4j instance for subsequent operations."""

    global _NEO4J_URI, _NEO4J_AUTH
    _NEO4J_URI = uri
    _NEO4J_AUTH = auth
    LOGGER.debug("Configured Neo4j URI=%s auth=%s", uri, "set" if auth else "none")


def load_events(run_id: str, events_file: Optional[Path] = None) -> List[Dict]:
    """Load events from the orchestrator run directory."""

    if events_file is None:
        events_file = DEFAULT_EVENTS_DIR / run_id / "events.json"

    try:
        with open(events_file, "r", encoding="utf-8") as handle:
            events = json.load(handle)
            if not isinstance(events, list):
                raise ValueError("Events file does not contain a list of events")
            LOGGER.info("Loaded %d events from %s", len(events), events_file)
            return events
    except FileNotFoundError as exc:
        LOGGER.error("Events file not found: %s", events_file)
        raise exc


def _normalize_event(event: Dict) -> Dict:
    """Ensure an event dictionary contains the expected keys."""

    agent = event.get("agent") or event.get("agent_id") or "unknown-agent"
    tactic = event.get("tactic") or event.get("mitre_tactic") or "unknown-tactic"
    technique = event.get("technique") or event.get("mitre_technique") or "unknown-technique"
    timestamp = event.get("timestamp") or event.get("time")
    event_id = event.get("event_id") or event.get("id")

    normalized = {
        "agent": str(agent),
        "tactic": str(tactic),
        "technique": str(technique),
        "timestamp": str(timestamp) if timestamp is not None else None,
        "event_id": str(event_id) if event_id is not None else None,
    }
    return normalized


def _merge_graph_entities(
    tx,
    agent_name: str,
    technique_name: str,
    tactic_name: str,
    timestamp: Optional[str],
    event_id: Optional[str],
) -> Tuple[int, int]:
    """Create or update graph entities for a single event."""

    query = """
    MERGE (agent:Agent {name: $agent_name})
    ON CREATE SET
        agent.event_ids = CASE WHEN $event_id IS NULL THEN [] ELSE [$event_id] END,
        agent.timestamps = CASE WHEN $timestamp IS NULL THEN [] ELSE [$timestamp] END
    ON MATCH SET
        agent.event_ids = CASE
            WHEN $event_id IS NULL OR agent.event_ids IS NULL OR $event_id IN agent.event_ids
                THEN agent.event_ids
            ELSE agent.event_ids + $event_id
        END,
        agent.timestamps = CASE
            WHEN $timestamp IS NULL OR agent.timestamps IS NULL OR $timestamp IN agent.timestamps
                THEN agent.timestamps
            ELSE agent.timestamps + $timestamp
        END

    MERGE (tech:Technique {name: $technique_name})
    ON CREATE SET
        tech.event_ids = CASE WHEN $event_id IS NULL THEN [] ELSE [$event_id] END,
        tech.timestamps = CASE WHEN $timestamp IS NULL THEN [] ELSE [$timestamp] END
    ON MATCH SET
        tech.event_ids = CASE
            WHEN $event_id IS NULL OR tech.event_ids IS NULL OR $event_id IN tech.event_ids
                THEN tech.event_ids
            ELSE tech.event_ids + $event_id
        END,
        tech.timestamps = CASE
            WHEN $timestamp IS NULL OR tech.timestamps IS NULL OR $timestamp IN tech.timestamps
                THEN tech.timestamps
            ELSE tech.timestamps + $timestamp
        END

    MERGE (tac:Tactic {name: $tactic_name})
    ON CREATE SET
        tac.event_ids = CASE WHEN $event_id IS NULL THEN [] ELSE [$event_id] END,
        tac.timestamps = CASE WHEN $timestamp IS NULL THEN [] ELSE [$timestamp] END
    ON MATCH SET
        tac.event_ids = CASE
            WHEN $event_id IS NULL OR tac.event_ids IS NULL OR $event_id IN tac.event_ids
                THEN tac.event_ids
            ELSE tac.event_ids + $event_id
        END,
        tac.timestamps = CASE
            WHEN $timestamp IS NULL OR tac.timestamps IS NULL OR $timestamp IN tac.timestamps
                THEN tac.timestamps
            ELSE tac.timestamps + $timestamp
        END

    MERGE (agent)-[r1:PERFORMS]->(tech)
    ON CREATE SET
        r1.event_ids = CASE WHEN $event_id IS NULL THEN [] ELSE [$event_id] END,
        r1.timestamps = CASE WHEN $timestamp IS NULL THEN [] ELSE [$timestamp] END
    ON MATCH SET
        r1.event_ids = CASE
            WHEN $event_id IS NULL OR r1.event_ids IS NULL OR $event_id IN r1.event_ids
                THEN r1.event_ids
            ELSE r1.event_ids + $event_id
        END,
        r1.timestamps = CASE
            WHEN $timestamp IS NULL OR r1.timestamps IS NULL OR $timestamp IN r1.timestamps
                THEN r1.timestamps
            ELSE r1.timestamps + $timestamp
        END

    MERGE (tech)-[r2:BELONGS_TO]->(tac)
    ON CREATE SET
        r2.event_ids = CASE WHEN $event_id IS NULL THEN [] ELSE [$event_id] END,
        r2.timestamps = CASE WHEN $timestamp IS NULL THEN [] ELSE [$timestamp] END
    ON MATCH SET
        r2.event_ids = CASE
            WHEN $event_id IS NULL OR r2.event_ids IS NULL OR $event_id IN r2.event_ids
                THEN r2.event_ids
            ELSE r2.event_ids + $event_id
        END,
        r2.timestamps = CASE
            WHEN $timestamp IS NULL OR r2.timestamps IS NULL OR $timestamp IN r2.timestamps
                THEN r2.timestamps
            ELSE r2.timestamps + $timestamp
        END
    """

    result = tx.run(
        query,
        agent_name=agent_name,
        technique_name=technique_name,
        tactic_name=tactic_name,
        timestamp=timestamp,
        event_id=event_id,
    )
    summary = result.consume()
    return summary.counters.nodes_created, summary.counters.relationships_created


def _get_driver(uri: Optional[str] = None, auth: Optional[Tuple[str, str]] = None):
    uri_to_use = uri or _NEO4J_URI
    auth_to_use = auth if auth is not None else _NEO4J_AUTH
    if auth_to_use:
        return GraphDatabase.driver(uri_to_use, auth=basic_auth(*auth_to_use))
    return GraphDatabase.driver(uri_to_use)


def build_attack_graph(events: List[Dict]) -> Tuple[int, int]:
    """Build an attack graph for the provided events.

    Returns a tuple containing the number of nodes and relationships created.
    """

    total_nodes = 0
    total_relationships = 0
    try:
        with _get_driver() as driver:
            with driver.session() as session:
                for event in events:
                    normalized = _normalize_event(event)
                    nodes_created, relationships_created = session.write_transaction(
                        _merge_graph_entities,
                        normalized["agent"],
                        normalized["technique"],
                        normalized["tactic"],
                        normalized["timestamp"],
                        normalized["event_id"],
                    )
                    total_nodes += nodes_created
                    total_relationships += relationships_created
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to build attack graph: %s", exc)
        raise

    LOGGER.info(
        "Attack graph build complete: %d nodes created, %d relationships created",
        total_nodes,
        total_relationships,
    )
    return total_nodes, total_relationships


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Neo4j attack graph from run events")
    parser.add_argument("--run-id", required=True, help="Identifier of the orchestrator run")
    parser.add_argument(
        "--events-file",
        type=Path,
        default=None,
        help="Explicit path to an events.json file (overrides --run-id path)",
    )
    parser.add_argument(
        "--neo4j",
        default=DEFAULT_NEO4J_URI,
        help="Bolt URI for the Neo4j instance (default bolt://neo4j:7687)",
    )
    parser.add_argument("--username", default=None, help="Neo4j username (optional)")
    parser.add_argument("--password", default=None, help="Neo4j password (optional)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default INFO)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    auth: Optional[Tuple[str, str]] = None
    if args.username and args.password:
        auth = (args.username, args.password)

    configure_neo4j(args.neo4j, auth)

    try:
        events = load_events(args.run_id, args.events_file)
    except Exception:  # noqa: BLE001
        return 1

    try:
        build_attack_graph(events)
    except Exception:  # noqa: BLE001
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
