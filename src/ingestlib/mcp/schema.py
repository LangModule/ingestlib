"""JSON Schema → Pydantic model, so an MCP agent can define an extract shape.

`extract()` takes a Pydantic class; an agent over MCP passes JSON. This turns
the agent-supplied JSON Schema into a dynamic Pydantic model the existing
`extract()` accepts — the whole reason extract works over MCP with no change
to the library core. Handles scalars, arrays (with item type), and nested
objects (recursively); every field's `description` flows into the model so it
reaches extract's prompt.
"""
from typing import Any

from pydantic import BaseModel, Field, create_model

_JSON_TO_PY: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}


def _py_type(spec: dict[str, Any], name: str) -> Any:
    """Map one JSON-Schema property spec to a Python type annotation."""
    kind = spec.get("type", "string")
    if kind == "array":
        items = spec.get("items") or {}
        return list[_py_type(items, f"{name}Item")] if items else list
    if kind == "object" and "properties" in spec:
        return model_from_json_schema(spec.get("title") or f"{name}Obj", spec)
    return _JSON_TO_PY.get(kind, str)


def model_from_json_schema(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Build a Pydantic model from a JSON-Schema object.

    Raises ValueError on anything that isn't an object-with-properties — the
    shape extract() needs. Required keys become required fields; the rest are
    Optional (default None).
    """
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        raise ValueError(
            "extract schema must be a JSON Schema object with a 'properties' map, "
            'e.g. {"type": "object", "properties": {"total": {"type": "number"}}}'
        )
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for field_name, spec in schema["properties"].items():
        spec = spec or {}
        py = _py_type(spec, field_name)
        desc = spec.get("description", "")
        if field_name in required:
            fields[field_name] = (py, Field(..., description=desc))
        else:
            fields[field_name] = (py | None, Field(default=None, description=desc))

    model_name = schema.get("title") or name or "ExtractSchema"
    # a valid Python identifier for the class name (extract keys artifacts by it)
    model_name = "".join(c if c.isalnum() else "_" for c in model_name) or "ExtractSchema"
    return create_model(model_name, **fields)
