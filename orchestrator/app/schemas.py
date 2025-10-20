"""Pydantic schemas for orchestrator API payload validation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, constr, root_validator

TechniqueId = constr(regex=r"^T\d{4}$")
ShortText = constr(max_length=280)


class SchemaVersion(str, Enum):
    """Enumeration of supported API schema versions."""

    REDOPS_V1 = "redops/v1"


class Tactic(str, Enum):
    """Permitted adversary tactics."""

    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial-access"
    EXECUTION = "execution"
    PRIVILEGE_ESCALATION = "privilege-escalation"
    LATERAL_MOVEMENT = "lateral-movement"
    CREDENTIAL_ACCESS = "credential-access"
    COMMAND_AND_CONTROL = "command-and-control"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    PERSISTENCE = "persistence"
    DISCOVERY = "discovery"
    UNKNOWN = "unknown"


class EventInV1(BaseModel):
    """Event payload reported by red team agents."""

    timestamp: datetime
    agent_id: str = Field(..., min_length=1)
    kind: str = Field("simulated", const=True)
    tactic: Tactic
    technique: Optional[TechniqueId] = None
    note: Optional[ShortText] = None

    class Config:
        extra = "allow"


class ResponseInV1(BaseModel):
    """Blue team response payload recorded against a run."""

    timestamp: datetime
    agent_id: str = Field(..., min_length=1)
    response: Optional[str] = None
    reason: Optional[ShortText] = None
    kind: Optional[str] = Field(None, regex=r"^simulated$")
    note: Optional[ShortText] = None
    apply_policy_changes: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"

    @root_validator
    def _ensure_action_or_policy(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get("response") and not values.get("apply_policy_changes"):
            raise ValueError("response or apply_policy_changes must be provided")
        return values


class Detection(BaseModel):
    """Detection record produced by lightweight analytics."""

    timestamp: datetime
    run_id: str
    type: str
    tactic: Optional[Tactic] = None
    source_event: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class VersionedEventIn(BaseModel):
    """Schema version envelope for event ingestion."""

    schema_version: SchemaVersion = Field(..., alias="schema")
    payload: EventInV1

    class Config:
        allow_population_by_field_name = True


class VersionedResponseIn(BaseModel):
    """Schema version envelope for blue team responses."""

    schema_version: SchemaVersion = Field(..., alias="schema")
    payload: ResponseInV1

    class Config:
        allow_population_by_field_name = True


class VersionedDetections(BaseModel):
    """Schema version envelope for detections responses."""

    schema_version: SchemaVersion = Field(..., alias="schema")
    payload: List[Detection]

    class Config:
        allow_population_by_field_name = True
