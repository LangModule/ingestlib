"""sources.yaml → Source instances, resolved by name and cached.

retrieve(sources=[...]) passes names; this turns each into a DocumentSource or
a SqlSource from its sources.yaml spec. Instances are cached (a SqlSource holds
an engine pool); reset_config() clears them via reset_registry().
"""
from ingestlib.config import SourceSpec, get_sources_config
from ingestlib.sources.base import Source


_cache: dict[str, Source] = {}


def resolve_sources(names: list[str]) -> list[Source]:
    """The Source instances for these sources.yaml names (built once, cached)."""
    return [_resolve_one(name) for name in names]


def _resolve_one(name: str) -> Source:
    source = _cache.get(name)
    if source is None:
        specs = get_sources_config().sources
        spec = specs.get(name)
        if spec is None:
            raise ValueError(
                f"unknown source {name!r} — declare it in sources.yaml "
                f"(configured: {sorted(specs) or 'none'})"
            )
        source = _build(spec)
        _cache[name] = source
    return source


def _build(spec: SourceSpec) -> Source:
    if spec.type == "documents":
        from ingestlib.sources.documents import DocumentSource

        return DocumentSource(spec.name, namespace=spec.namespace)
    from ingestlib.sources.sql.source import SqlSource

    return SqlSource(spec)


def reset_registry() -> None:
    """Drop cached Source instances (e.g. after a config edit)."""
    _cache.clear()
