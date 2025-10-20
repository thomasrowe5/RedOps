import asyncio
import json
from datetime import datetime as _real_datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.app import jobs
from orchestrator.app.jobs import BatchWriter, JobBus


def test_queue_backpressure_and_flush(tmp_path: Path):
    asyncio.run(_run_queue_backpressure_and_flush(tmp_path))


async def _run_queue_backpressure_and_flush(tmp_path: Path) -> None:
    bus = JobBus(maxsize=5, default_timeout=0.05)
    run_id = "t1"

    for i in range(5):
        assert await bus.put_event(run_id, {"i": i})

    # Next put should fail because the queue is full and the timeout elapses.
    assert not await bus.put_event(run_id, {"i": 99}, timeout=0.01)

    generator = bus.get_event(run_id)
    drained = []
    for _ in range(3):
        drained.append(await generator.__anext__())

    writer = BatchWriter(bus, tmp_path / "data" / "runs")
    max_bytes = writer.MAX_MB * 1024 * 1024
    writer._write_batch(run_id, drained, max_bytes)

    ndjson_path = tmp_path / "data" / "runs" / run_id / "events.ndjson"
    assert ndjson_path.exists()
    content = ndjson_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 3
    assert [json.loads(line) for line in content] == drained

    # Queue should now have space for new events after draining.
    assert await bus.put_event(run_id, {"i": 100})

    await generator.aclose()


def test_atomic_rotate(tmp_path: Path, monkeypatch):
    asyncio.run(_run_atomic_rotate(tmp_path, monkeypatch))


async def _run_atomic_rotate(tmp_path: Path, monkeypatch) -> None:
    bus = JobBus()
    writer = BatchWriter(bus, tmp_path / "data" / "runs")
    run_id = "t2"
    run_dir = tmp_path / "data" / "runs" / run_id
    run_dir.mkdir(parents=True)

    path = run_dir / "events.ndjson"
    legacy_path = path.with_suffix(".json")
    path.write_text("x" * 128, encoding="utf-8")
    legacy_path.write_text("y" * 128, encoding="utf-8")

    class _FixedDatetime(_real_datetime):
        @classmethod
        def utcnow(cls):
            return cls(2024, 1, 2, 3, 4, 5)

    monkeypatch.setattr(jobs, "datetime", _FixedDatetime)

    events = [{"i": i} for i in range(2)]
    payload = 20  # bytes per event is small; force rotation via tiny max
    writer._write_batch(run_id, events, max_bytes=payload)

    rotated_name = "events-20240102030405.ndjson"
    rotated_legacy = "events-20240102030405.json"

    assert (run_dir / rotated_name).exists()
    assert (run_dir / rotated_legacy).exists()
    assert not any(f.suffix.endswith(".tmp") for f in run_dir.glob("*"))

    new_payload = (run_dir / "events.ndjson").read_text(encoding="utf-8")
    assert new_payload.strip().splitlines()
    assert json.loads(new_payload.strip().splitlines()[0]) == {"i": 0}
