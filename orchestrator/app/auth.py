"""Authentication helpers for the RedOps orchestrator."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import Depends, Header, HTTPException, status

ALLOWED_ROLES = {"agent_red", "agent_blue", "operator"}


@dataclass
class Actor:
    """Represents an authenticated caller of the orchestrator."""

    name: str
    role: str


class AuthError(Exception):
    """Raised when a JWT fails validation."""


def _get_secret() -> bytes:
    secret = os.getenv("REDOPS_JWT_SECRET")
    if not secret:
        raise RuntimeError("REDOPS_JWT_SECRET is not configured")
    return secret.encode("utf-8")


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(message: bytes, secret: bytes) -> bytes:
    return hmac.new(secret, message, hashlib.sha256).digest()


def create_token(actor: str, role: str, *, expires_in: int = 3600) -> str:
    """Create a signed HS256 JWT representing the given actor."""

    if role not in ALLOWED_ROLES:
        raise ValueError(f"Unsupported role: {role}")

    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {"sub": actor, "role": role, "iat": now, "exp": now + expires_in}

    header_segment = _b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_segment = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    secret = _get_secret()
    signature_segment = _b64encode(_sign(signing_input, secret))
    return f"{header_segment}.{payload_segment}.{signature_segment}"


def _decode_token(token: str) -> Actor:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise AuthError("invalid token format") from exc

    secret = _get_secret()
    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    expected_signature = _sign(signing_input, secret)
    try:
        provided_signature = _b64decode(signature_segment)
    except (binascii.Error, ValueError) as exc:
        raise AuthError("invalid token signature") from exc

    if not hmac.compare_digest(expected_signature, provided_signature):
        raise AuthError("signature mismatch")

    try:
        payload: dict[str, Any] = json.loads(_b64decode(payload_segment))
    except (json.JSONDecodeError, ValueError) as exc:
        raise AuthError("invalid token payload") from exc

    role = payload.get("role")
    subject = payload.get("sub")
    expiration = payload.get("exp")

    if role not in ALLOWED_ROLES or not isinstance(subject, str):
        raise AuthError("invalid actor")
    if not isinstance(expiration, int) or expiration < int(time.time()):
        raise AuthError("token expired")

    return Actor(name=subject, role=role)


def get_current_actor(authorization: str = Header(default="")) -> Actor:
    """FastAPI dependency that extracts the caller identity from a JWT."""

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")

    token = authorization.split(" ", 1)[1]
    try:
        actor = _decode_token(token)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    return actor


def require_roles(*roles: str) -> Callable[[Actor], Actor]:
    """Return a dependency that ensures the caller has one of the allowed roles."""

    invalid_roles = set(roles) - ALLOWED_ROLES
    if invalid_roles:
        raise ValueError(f"Unsupported roles requested: {sorted(invalid_roles)}")

    async def dependency(actor: Actor = Depends(get_current_actor)) -> Actor:
        if actor.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient permissions")
        return actor

    return dependency


__all__ = ["Actor", "create_token", "get_current_actor", "require_roles", "ALLOWED_ROLES"]
