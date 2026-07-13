"""Root-level fixtures shared across every test subfolder."""
import numpy as np
import pytest


@pytest.fixture(scope="session")
def cos_sim():
    def _cos(a, b) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    return _cos
