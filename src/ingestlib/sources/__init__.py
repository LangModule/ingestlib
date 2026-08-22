"""Structured + document sources for retrieve(sources=[...]).

A Source is a queryable backend — the document corpus or a SQL database — that
turns a question into normalized SourceResults. Sources are declared in a
sources.yaml sidecar (see sources.example.yaml); SQL sources need their pip
extra, e.g. `pip install "ingestlib[postgres]"`.
"""
from ingestlib.sources.base import Source, SourceResult, SourceType

__all__ = ["Source", "SourceResult", "SourceType"]
