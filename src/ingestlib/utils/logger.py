"""Central logger for ingestlib.

Every module gets its logger via:

    from ingestlib.utils.logger import get_logger
    logger = get_logger(__name__)

Configuration is applied once at package import (see ingestlib/__init__.py) and
can be re-applied at any time by calling configure().

Environment variables:
    INGESTLIB_LOG_LEVEL        — DEBUG | INFO | WARNING | ERROR (default INFO)
    INGESTLIB_LOG_THIRD_PARTY  — "1" raises third-party loggers to the same level
    INGESTLIB_LOG_COLOR        — "0" disables colored output (default: auto via TTY)
"""
import logging
import os
import sys

_ROOT_LOGGER_NAME = "ingestlib"
_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)-42s %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"

# Noisy third-party loggers. Quieted to WARNING by default so ingestlib's own
# logs stay readable; configure(include_third_party=True) raises them instead.
_THIRD_PARTY_LOGGERS: tuple[str, ...] = (
    "paddleocr",
    "paddlex",
    "ppocr",
    "openai",
    "httpx",
    "httpcore",
    "botocore",
)

_LEVEL_COLORS = {
    logging.DEBUG: "\033[2m",       # dim
    logging.INFO: "\033[32m",       # green
    logging.WARNING: "\033[33m",    # yellow
    logging.ERROR: "\033[31m",      # red
    logging.CRITICAL: "\033[1;31m", # bold red
}
_RESET = "\033[0m"
_DIM = "\033[2m"


class _ColorFormatter(logging.Formatter):
    """Colors the level name and dims the timestamp/logger name on TTY output."""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        ts = self.formatTime(record, _DEFAULT_DATEFMT)
        level = f"{color}{record.levelname:<8}{_RESET}"
        name = f"{_DIM}{record.name:<42}{_RESET}"
        line = f"{_DIM}{ts}{_RESET} {level} {name} {record.getMessage()}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def get_logger(name: str) -> logging.Logger:
    """Return a logger for `name` (typically pass __name__).

    Loggers under the `ingestlib.*` namespace inherit the handler and level
    configured on the ingestlib root logger.
    """
    return logging.getLogger(name)


def _use_color() -> bool:
    if os.environ.get("INGESTLIB_LOG_COLOR") == "0":
        return False
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def configure(
    level: int | str = logging.INFO,
    *,
    include_third_party: bool = False,
) -> None:
    """Attach a stderr StreamHandler to the ingestlib logger.

    Idempotent. Safe to call multiple times — old handlers are removed first
    so log lines never duplicate.

    Third-party loggers (paddlex, httpx, botocore, ...) are set to WARNING so
    their INFO chatter doesn't drown ingestlib's own logs. Pass
    include_third_party=True to raise them to `level` instead (debugging aid).

    Args:
        level: numeric or string level ("DEBUG", "INFO", "WARNING", ...).
        include_third_party: if True, third-party loggers follow `level`.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)

    # Clear pre-existing handlers so reconfiguration doesn't duplicate lines.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    if _use_color():
        handler.setFormatter(_ColorFormatter())
    else:
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)

    # Prevent double-output through Python's root logger.
    root.propagate = False

    third_party_level = level if include_third_party else logging.WARNING
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)


def _auto_configure() -> None:
    """Configure at package import.

    Level comes from INGESTLIB_LOG_LEVEL if set, else INFO by default.
    Third-party inclusion follows INGESTLIB_LOG_THIRD_PARTY=1 if set.
    """
    level = os.environ.get("INGESTLIB_LOG_LEVEL", "INFO")
    include_third_party = os.environ.get("INGESTLIB_LOG_THIRD_PARTY") == "1"
    configure(level=level, include_third_party=include_third_party)
