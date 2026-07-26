"""ingestlib — self-hosted document intelligence library."""
from importlib.metadata import PackageNotFoundError, version

from ingestlib.utils.logger import _auto_configure as _auto_configure_logger

try:
    __version__ = version("ingestlib")
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0.0.0.dev0"

_auto_configure_logger()
