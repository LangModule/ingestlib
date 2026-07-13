"""Cross-cutting utilities for ingestlib.

Public modules:
    logger — get_logger(name), configure(level, include_third_party)
             env: INGESTLIB_LOG_LEVEL, INGESTLIB_LOG_THIRD_PARTY, INGESTLIB_LOG_COLOR
    files  — sha256_of_file(path) → the content-derived document ID
"""
