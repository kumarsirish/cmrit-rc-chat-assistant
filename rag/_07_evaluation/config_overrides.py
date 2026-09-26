"""Lets helper_scripts/run_experiment.py change a config.yaml value for one
run without editing the file, e.g. "--set retrieval.top_k=5".

AppConfig is a plain nested dataclass tree (see rag/common/config.py),
so this just walks attribute names and uses setattr — no schema of
its own to keep in sync.
"""

import dataclasses


def apply_overrides(config, overrides: list):
    """overrides: list of "dotted.path=value" strings. Mutates config
    in place and returns it, for convenient chaining."""
    for override in overrides:
        path, separator, raw_value = override.partition("=")
        if not separator:
            raise ValueError(f"Invalid override {override!r}, expected key.path=value")
        _set_by_path(config, path.strip(), raw_value.strip())
    return config


def _set_by_path(config, path: str, raw_value: str) -> None:
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Unknown config path {path!r}: no field named {part!r}")
        target = getattr(target, part)

    field_name = parts[-1]
    if not hasattr(target, field_name):
        raise ValueError(f"Unknown config path {path!r}: no field named {field_name!r}")

    current_value = getattr(target, field_name)
    setattr(target, field_name, _parse_value(raw_value, type(current_value)))


def _parse_value(raw_value: str, target_type: type):
    # Matches the existing field's type, so "5" becomes an int if the
    # field is already an int, "0.7" becomes a float, etc.
    if target_type is bool:
        return raw_value.lower() in ("1", "true", "yes")
    if target_type is int:
        return int(raw_value)
    if target_type is float:
        return float(raw_value)
    return raw_value


def config_to_dict(config) -> dict:
    """Turns AppConfig (nested dataclasses) into a plain dict, for
    logging as an MLflow artifact."""
    return dataclasses.asdict(config)
