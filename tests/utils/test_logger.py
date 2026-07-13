"""Logger configuration behavior — pure logic, always run."""
import logging

from ingestlib.utils.logger import _THIRD_PARTY_LOGGERS, configure, get_logger


def test_get_logger_namespaces_under_ingestlib():
    lg = get_logger("ingestlib.operations.parse.pipeline")
    assert lg.name == "ingestlib.operations.parse.pipeline"


def test_configure_is_idempotent_no_duplicate_handlers():
    configure(level="INFO")
    configure(level="INFO")
    configure(level="INFO")
    root = logging.getLogger("ingestlib")
    assert len(root.handlers) == 1, "reconfiguring must not stack handlers"


def test_configure_sets_level_from_string():
    configure(level="DEBUG")
    assert logging.getLogger("ingestlib").level == logging.DEBUG
    configure(level="WARNING")
    assert logging.getLogger("ingestlib").level == logging.WARNING


def test_unknown_level_string_falls_back_to_info():
    configure(level="NOT_A_LEVEL")
    assert logging.getLogger("ingestlib").level == logging.INFO


def test_third_party_loggers_quieted_by_default():
    configure(level="DEBUG")
    for name in _THIRD_PARTY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


def test_third_party_loggers_follow_level_when_opted_in():
    configure(level="DEBUG", include_third_party=True)
    for name in _THIRD_PARTY_LOGGERS:
        assert logging.getLogger(name).level == logging.DEBUG
    # restore the default policy for any tests that run after this one
    configure(level="INFO")


def test_no_propagation_to_python_root_logger():
    configure(level="INFO")
    assert logging.getLogger("ingestlib").propagate is False
