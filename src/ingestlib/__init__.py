"""ingestlib — self-hosted document intelligence library."""
from importlib.metadata import PackageNotFoundError, version

from ingestlib.utils.logger import _auto_configure as _auto_configure_logger

try:
    __version__ = version("ingestlib")
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0.0.0.dev0"

_auto_configure_logger()


# Convenience shorthands — ingestlib.retrieve(...) / ingestlib.ingest(...) —
# resolved lazily so `import ingestlib` stays filesystem-free (the services
# layer pulls in config + connectors, which must not import at package import).
_LAZY_SERVICES = frozenset({"retrieve", "aretrieve", "ingest", "aingest"})


def __getattr__(name: str):
    if name in _LAZY_SERVICES:
        from ingestlib import services

        return getattr(services, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
