"""Connector SDKs are pip extras — storage must not import one until it's
actually selected, and a missing SDK must name the exact extra to install.

The isolation tests run in subprocesses: laziness is a property of a fresh
interpreter, and 'SDK not installed' is made REAL by blocking the module in
that process, not by mocking an importer in ours.
"""
import subprocess
import sys

_SDKS = (
    "pinecone", "qdrant_client", "fastembed", "psycopg", "pgvector",
    "pymongo", "pymilvus", "opensearchpy", "weaviate",
)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )


def test_core_import_touches_no_connector_sdk():
    result = _run(
        "import sys\n"
        "from ingestlib.storage import SqliteStore, VectorStore, RetrievedChunk\n"
        f"loaded = [m for m in {_SDKS!r} if m in sys.modules]\n"
        "assert not loaded, f'connector SDKs imported eagerly: {loaded}'\n"
        "print('clean')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_selected_connector_loads_on_demand():
    from ingestlib.storage import QdrantStore
    from ingestlib.storage.base import VectorStore

    assert issubclass(QdrantStore, VectorStore)


def test_missing_sdk_names_the_extra():
    """With pymilvus genuinely absent from the process, MilvusStore must
    raise the pip-extra instruction, not a bare ModuleNotFoundError."""
    result = _run(
        "import sys, importlib.abc\n"
        "class Absent(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'pymilvus' or name.startswith('pymilvus.'):\n"
        "            raise ModuleNotFoundError(f'No module named {name!r}', name=name)\n"
        "sys.meta_path.insert(0, Absent())\n"
        "try:\n"
        "    from ingestlib.storage import MilvusStore\n"
        "except ImportError as exc:\n"
        "    assert 'ingestlib[milvus]' in str(exc), str(exc)\n"
        "    print('hinted')\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "hinted" in result.stdout


def test_unknown_attribute_still_raises_attribute_error():
    import ingestlib.storage as storage
    import pytest

    with pytest.raises(AttributeError, match="ChromaStore"):
        storage.ChromaStore
