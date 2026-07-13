"""IngestResult behavior — pure, always run."""
import pytest

from ingestlib.services.ingest.models import IngestResult


def test_total_seconds_sums_stage_durations():
    r = IngestResult(status="ingested", doc_id="d",
                     durations={"parse": 10.0, "embed": 0.5, "upsert": 1.5})
    assert r.total_seconds == 12.0


def test_defaults_and_frozen():
    r = IngestResult(status="skipped", doc_id="d")
    assert r.chunks == 0 and r.durations == {} and r.total_seconds == 0.0
    with pytest.raises(Exception):
        r.status = "x"  # type: ignore[misc]
