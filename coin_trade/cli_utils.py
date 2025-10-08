from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def parse_key_value(pairs: Iterable[str] | None) -> dict[str, Any]:
    """Parse CLI overrides in ``key=value`` form into a dictionary."""
    result: dict[str, Any] = {}
    if not pairs:
        return result

    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Override '{raw}' must be in key=value format")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()

        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            parsed = _coerce_scalar(value)
        result[key] = parsed
    return result


def _coerce_scalar(value: str) -> Any:
    """Attempt numeric coercion fallback to raw string."""
    try:
        if "." in value:
            num = float(value)
            return int(num) if float(num).is_integer() else num
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_structured(path: str | None) -> Any | None:
    """Load JSON/YAML configuration files."""
    if path is None:
        return None
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to read YAML files. Install with 'pip install pyyaml'."
            )
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config format for {path}")
