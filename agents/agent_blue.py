"""Blue team agent CLI tool for simulated response orchestration."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import yaml


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclass
class PolicyConfig:
    thresholds: Dict[str, int]
    actions: Dict[str, List[str]]
    cooldown_seconds: int


class CooldownTracker:
    def __init__(self, cooldown_seconds: int) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_triggered: Dict[str, float] = {}

    def ready(self, action: str, now: float) -> bool:
        last = self._last_triggered.get(action)
        return last is None or (now - last) >= self.cooldown_seconds

    def mark(self, action: str, now: float) -> None:
        self._last_triggered[action] = now


class BlueAgent:
    def __init__(
        self,
        orchestrator_base: str,
        run_id: str,
        agent_id: str,
        policy_config: PolicyConfig,
        poll_interval: float,
    ) -> None:
        self.session = requests.Session()
        self.orchestrator_base = orchestrator_base.rstrip("/")
        self.run_id = run_id
        self.agent_id = agent_id
        self.policy_config = policy_config
        self.poll_interval = poll_interval
        self.cooldowns = CooldownTracker(policy_config.cooldown_seconds)
        self.detection_counts: Dict[str, int] = defaultdict(int)
        self.seen_detection_ids: set[str] = set()
        self.last_detection_ts: Optional[str] = None
        self.current_policy: Optional[Dict[str, Any]] = None

    # -------------------------- Utility helpers -------------------------- #
    def _run_url(self, path: str) -> str:
        return f"{self.orchestrator_base}/runs/{self.run_id}{path}"

    def _parse_iso_ts(self, value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            if value.endswith("Z"):
                # Support fractional seconds
                try:
                    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
                except ValueError:
                    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
                return parsed.replace(tzinfo=timezone.utc)
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            return None

    # -------------------------- Network helpers ------------------------- #
    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[int, Optional[Any]]:
        try:
            response = self.session.get(url, timeout=5, params=params)
            if response.status_code == 200:
                return 200, response.json()
            return response.status_code, None
        except requests.RequestException as exc:
            print(f"[blue-agent] GET {url} failed: {exc}", file=sys.stderr)
            return 0, None

    def _post_json(self, url: str, payload: Dict[str, Any]) -> bool:
        try:
            response = self.session.post(url, json=payload, timeout=5)
            if response.status_code in (200, 201, 202, 204):
                return True
            print(
                f"[blue-agent] POST {url} failed with status {response.status_code}: {response.text}",
                file=sys.stderr,
            )
        except requests.RequestException as exc:
            print(f"[blue-agent] POST {url} failed: {exc}", file=sys.stderr)
        return False

    # -------------------------- Detection polling ----------------------- #
    def poll_detections(self) -> List[Dict[str, Any]]:
        params = {}
        if self.last_detection_ts:
            params["since_ts"] = self.last_detection_ts
        detections_url = self._run_url("/detections")
        status, payload = self._get_json(detections_url, params=params)
        if status == 200 and isinstance(payload, list):
            return payload
        if status == 404:
            return self._read_detections_from_disk(params.get("since_ts"))
        return []

    def _read_detections_from_disk(self, since_ts: Optional[str]) -> List[Dict[str, Any]]:
        path = Path("orchestrator/data/runs") / self.run_id / "detections.json"
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[blue-agent] Failed to read detections file {path}: {exc}", file=sys.stderr)
            return []
        if not isinstance(payload, list):
            return []
        if not since_ts:
            return payload
        threshold = self._parse_iso_ts(since_ts)
        if threshold is None:
            return payload
        results: List[Dict[str, Any]] = []
        for item in payload:
            timestamp = item.get("timestamp")
            parsed = self._parse_iso_ts(timestamp) if isinstance(timestamp, str) else None
            if parsed and parsed > threshold:
                results.append(item)
        return results

    # -------------------------- Policy helpers -------------------------- #
    def _tactic_key(self, detection: Dict[str, Any]) -> Optional[str]:
        tactic = detection.get("tactic") or detection.get("attack_tactic")
        if not isinstance(tactic, str):
            return None
        tactic = tactic.strip().lower()
        if tactic in ("exfiltration", "collection", "command and control"):
            return "burst_uploads"
        if tactic in ("lateral movement", "credential access"):
            return "lateral_moves"
        if tactic in ("privilege escalation", "defense evasion"):
            return "priv_esc_detected"
        return None

    def _actions_for_detection(self, detection: Dict[str, Any]) -> List[str]:
        tactic = detection.get("tactic") or detection.get("attack_tactic")
        if not isinstance(tactic, str):
            return []
        tactic = tactic.strip().lower()
        action_map = {
            "exfiltration": self.policy_config.actions.get("on_exfil_detected", []),
            "collection": self.policy_config.actions.get("on_exfil_detected", []),
            "command and control": self.policy_config.actions.get("on_exfil_detected", []),
            "lateral movement": self.policy_config.actions.get("on_lateral_detected", []),
            "credential access": self.policy_config.actions.get("on_lateral_detected", []),
            "privilege escalation": self.policy_config.actions.get("on_priv_esc_detected", []),
            "defense evasion": self.policy_config.actions.get("on_priv_esc_detected", []),
        }
        return action_map.get(tactic, [])

    def _policy_changes_for_action(self, action: str, detection: Dict[str, Any]) -> Dict[str, Any]:
        if action == "block_egress":
            return {"net_egress_block": True}
        if action == "rotate_creds":
            return {"credential_rotation": True}
        if action == "isolate_service":
            service = detection.get("asset") or detection.get("host") or "unknown-service"
            return {"service_isolation": [service]}
        if action == "reissue_tokens":
            return {"token_reissue": True}
        if action == "increase_logging":
            return {"logging_level": "elevated"}
        return {}

    # ----------------------------- Processing --------------------------- #
    def process_detections(self, detections: Iterable[Dict[str, Any]]) -> None:
        new_detections = []
        for detection in detections:
            det_id = detection.get("id") or detection.get("detection_id")
            if det_id is not None:
                det_key = str(det_id)
                if det_key in self.seen_detection_ids:
                    continue
                self.seen_detection_ids.add(det_key)
            timestamp = detection.get("timestamp")
            if isinstance(timestamp, str):
                parsed_ts = self._parse_iso_ts(timestamp)
                last_parsed = self._parse_iso_ts(self.last_detection_ts) if self.last_detection_ts else None
                if parsed_ts and (last_parsed is None or parsed_ts > last_parsed):
                    self.last_detection_ts = timestamp
            new_detections.append(detection)

        for detection in new_detections:
            tactic_key = self._tactic_key(detection)
            if tactic_key:
                self.detection_counts[tactic_key] += 1
            actions = self._actions_for_detection(detection)
            if not actions:
                continue
            raw_reason = (
                detection.get("id")
                or detection.get("detection_id")
                or detection.get("tactic")
                or detection.get("attack_tactic")
                or "unknown"
            )
            reason = str(raw_reason)
            now = time.time()
            triggered_actions = []
            for action in actions:
                if not self.cooldowns.ready(action, now):
                    continue
                if not self._threshold_allows_action(tactic_key):
                    continue
                if self._emit_response(action, reason, detection):
                    self.cooldowns.mark(action, now)
                    triggered_actions.append(action)
            if triggered_actions:
                print(f"[blue-agent] Applied actions {triggered_actions} for reason {reason}")

    def _threshold_allows_action(self, tactic_key: Optional[str]) -> bool:
        if not tactic_key:
            return True
        threshold = self.policy_config.thresholds.get(tactic_key, 1)
        return self.detection_counts[tactic_key] >= threshold

    def _emit_response(self, action: str, reason: str, detection: Dict[str, Any]) -> bool:
        payload = {
            "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).strftime(ISO_FORMAT),
            "agent_id": self.agent_id,
            "response": action,
            "reason": reason,
            "kind": "simulated",
            "note": "blue-response",
        }
        responses_url = self._run_url("/responses")
        if not self._post_json(responses_url, payload):
            return False
        policy_changes = self._policy_changes_for_action(action, detection)
        if policy_changes:
            change_payload = {
                "timestamp": payload["timestamp"],
                "agent_id": self.agent_id,
                "apply_policy_changes": policy_changes,
            }
            self._post_json(responses_url, change_payload)
        return True

    # ------------------------------- Loop -------------------------------- #
    def run(self) -> None:
        print(f"[blue-agent] Starting Blue agent {self.agent_id} for run {self.run_id}")
        self._hydrate_policy_state()
        try:
            while True:
                detections = self.poll_detections()
                if detections:
                    self.process_detections(detections)
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("[blue-agent] Stopping agent")

    def _hydrate_policy_state(self) -> None:
        policy_url = self._run_url("/policy")
        status, payload = self._get_json(policy_url)
        if status == 200 and isinstance(payload, dict):
            self.current_policy = payload
            print("[blue-agent] Loaded current policy from orchestrator")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blue response agent")
    parser.add_argument("--orchestrator", required=True, help="Base URL for the orchestrator service")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--agent-id", required=True, help="Agent identifier")
    parser.add_argument("--policy-config", required=True, help="Path to blue policy YAML configuration")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds")
    return parser.parse_args(argv)


def validate_orchestrator_base(orchestrator_base: str) -> None:
    parsed = urlparse(orchestrator_base)
    host = parsed.hostname
    if not host or not (host == "localhost" or host.startswith("127.")):
        raise SystemExit("[blue-agent] Safety check failed: orchestrator must be localhost/127.x")


def load_policy_config(path: str) -> PolicyConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    thresholds = data.get("thresholds") or {}
    actions = data.get("actions") or {}
    cooldown_seconds = data.get("cooldown_seconds", 8)
    return PolicyConfig(
        thresholds={str(k): int(v) for k, v in thresholds.items()},
        actions={str(k): list(v) for k, v in actions.items()},
        cooldown_seconds=int(cooldown_seconds),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    validate_orchestrator_base(args.orchestrator)
    policy_config = load_policy_config(args.policy_config)
    agent = BlueAgent(
        orchestrator_base=args.orchestrator,
        run_id=args.run_id,
        agent_id=args.agent_id,
        policy_config=policy_config,
        poll_interval=args.poll_interval,
    )
    agent.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
