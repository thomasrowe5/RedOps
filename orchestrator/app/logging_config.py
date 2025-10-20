"""Logging configuration for the RedOps orchestrator."""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

try:
    import structlog
    from structlog.contextvars import get_contextvars, merge_contextvars
    from structlog.stdlib import BoundLogger, LoggerFactory, ProcessorFormatter
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    structlog = None  # type: ignore[assignment]
    get_contextvars = merge_contextvars = None  # type: ignore[assignment]
    BoundLogger = LoggerFactory = ProcessorFormatter = None  # type: ignore[assignment]


if structlog is not None:
    _TIMESTAMER = structlog.processors.TimeStamper(fmt="iso", key="timestamp")
else:  # pragma: no cover - fallback configuration
    _TIMESTAMER = None


def _add_run_and_agent(
    _: BoundLogger, __: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Ensure ``run_id`` and ``agent_id`` are present in every log entry."""

    if get_contextvars is None:
        return event_dict

    context = get_contextvars()
    if "run_id" not in event_dict or event_dict["run_id"] is None:
        event_dict["run_id"] = context.get("run_id")
    if "agent_id" not in event_dict or event_dict["agent_id"] is None:
        event_dict["agent_id"] = context.get("agent_id")
    if event_dict["run_id"] is None:
        event_dict["run_id"] = None
    if event_dict["agent_id"] is None:
        event_dict["agent_id"] = None
    return event_dict


def _configure_structlog() -> None:
    """Configure structlog to emit JSON events with contextual data."""

    if structlog is None:
        return

    structlog.configure(
        processors=[
            merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.processors.add_log_level,
            _TIMESTAMER,
            _add_run_and_agent,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=LoggerFactory(),
        wrapper_class=BoundLogger,
        cache_logger_on_first_use=True,
    )


def _build_formatter() -> logging.Formatter:
    if structlog is None:
        return logging.Formatter("%(message)s")

    return ProcessorFormatter(
        foreign_pre_chain=[
            merge_contextvars,
            structlog.processors.add_log_level,
            _TIMESTAMER,
            _add_run_and_agent,
        ],
        processors=[
            ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )


_configured = False


def configure_uvicorn() -> None:
    """Configure uvicorn loggers to emit structured JSON log events."""

    global _configured
    if _configured:
        return

    if structlog is None:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)
        for logger_name in ("uvicorn", "uvicorn.access"):
            logger = logging.getLogger(logger_name)
            logger.propagate = True
            if logger.level < logging.INFO:
                logger.setLevel(logging.INFO)
        _configured = True
        return

    _configure_structlog()
    formatter = _build_formatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

    for logger_name in ("uvicorn", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers = [handler]
        logger.propagate = False
        if logger.level < logging.INFO:
            logger.setLevel(logging.INFO)

    _configured = True
