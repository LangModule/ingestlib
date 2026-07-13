"""Root-level fixtures shared across every test subfolder."""
import numpy as np
import pytest

from ingestlib.config import get_config

# Config loads lazily — materialize it here so .env lands in os.environ
# before collection-time gates (e.g. the JINA_API_KEY skipif) are evaluated.
get_config()


@pytest.fixture(scope="session")
def cos_sim():
    def _cos(a, b) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    return _cos
