"""on_stage progress machinery (_stage / _notify) — pure, always run.

The full-pipeline event sequence is asserted in the services e2e suite; these
tests pin the contract the pieces guarantee: start/done ordering, durations,
callback bugs never killing an ingest, and failure attribution via an
unmatched "start".
"""
import pytest

from ingestlib.services import StageCallback
from ingestlib.services.ingest.ingestor import _notify, _stage


def test_stage_reports_start_done_and_records_duration():
    events: list[tuple[str, str]] = []
    durations: dict[str, float] = {}
    with _stage("parse", durations, lambda s, e: events.append((s, e))):
        pass
    assert events == [("parse", "start"), ("parse", "done")]
    assert durations["parse"] >= 0.0


def test_buggy_callback_never_kills_the_stage():
    def boom(stage: str, event: str) -> None:
        raise RuntimeError("bug in the caller's callback")

    durations: dict[str, float] = {}
    with _stage("embed", durations, boom):
        pass  # must not raise
    assert "embed" in durations, "the stage must complete and record its duration"


def test_failing_stage_leaves_start_unmatched_and_no_duration():
    events: list[str] = []
    durations: dict[str, float] = {}
    with pytest.raises(ValueError, match="stage blew up"):
        with _stage("split", durations, lambda s, e: events.append(e)):
            raise ValueError("stage blew up")
    assert events == ["start"], "the failed stage is the one whose start has no done"
    assert "split" not in durations


def test_none_callback_is_a_noop():
    durations: dict[str, float] = {}
    with _stage("upsert", durations, None):
        pass
    assert "upsert" in durations
    _notify(None, "parse", "start")  # must not raise


def test_stage_callback_type_is_exported():
    assert StageCallback is not None
