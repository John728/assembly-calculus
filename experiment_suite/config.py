from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class ExperimentCondition:
    list_type: str
    N: int = 0
    num_train_lists: int = 0
    num_test_lists: int = 0
    k_train_min: int = 1
    k_train_max: int = 1
    k_test_min: int = 1
    k_test_max: int = 1


@dataclass(frozen=True)
class ModelConfig:
    family: str
    values: dict[str, object]

    @property
    def model_name(self) -> str:
        value = self.values.get("model_name")
        return str(value) if value is not None else self.family


@dataclass(frozen=True)
class SuiteConfig:
    suite_name: str
    output_dir: str
    seeds: list[int]
    conditions: list[ExperimentCondition]
    models: dict[str, list[ModelConfig]]
    trace_plots: dict[str, object] | None
    config_path: str


def _to_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Expected {field_name} to be int-like, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.strip() == "":
            raise ValueError(f"Expected {field_name} to be int-like, got empty string")
        return int(value)
    raise ValueError(f"Expected {field_name} to be int-like, got {type(value).__name__}")


def _to_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected {field_name} to be str, got {type(value).__name__}")
    return value


def _condition_from_mapping(mapping: dict[str, object]) -> ExperimentCondition:
    return ExperimentCondition(
        list_type=_to_str(mapping["list_type"], "list_type"),
        N=_to_int(mapping.get("N", 0), "N"),
        num_train_lists=_to_int(mapping.get("num_train_lists", 0), "num_train_lists"),
        num_test_lists=_to_int(mapping.get("num_test_lists", 0), "num_test_lists"),
        k_train_min=_to_int(mapping.get("k_train_min", 1), "k_train_min"),
        k_train_max=_to_int(mapping.get("k_train_max", 1), "k_train_max"),
        k_test_min=_to_int(mapping.get("k_test_min", 1), "k_test_min"),
        k_test_max=_to_int(mapping.get("k_test_max", 1), "k_test_max"),
    )


def _string_key_dict(mapping: dict[object, object]) -> dict[str, object]:
    return {str(key): value for key, value in mapping.items()}


def _expect_type(value: object, expected_type: type, field_name: str) -> None:
    if not isinstance(value, expected_type):
        raise ValueError(f"Expected {field_name} to be {expected_type.__name__}, got {type(value).__name__}")


def _parse_scalar(raw: str) -> object:
    value = raw.strip()
    if value == "":
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if inner == "":
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
        return ast.literal_eval(value)
    return value


def _clean_lines(text: str) -> list[tuple[int, str]]:
    cleaned: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        cleaned.append((indent, raw_line.strip()))
    return cleaned


def _parse_block(lines: list[tuple[int, str]], start: int, indent: int) -> tuple[object, int]:
    if start >= len(lines):
        return {}, start
    _, content = lines[start]
    if content.startswith("- "):
        return _parse_list(lines, start, indent)
    return _parse_dict(lines, start, indent)


def _parse_dict(lines: list[tuple[int, str]], start: int, indent: int) -> tuple[dict[str, object], int]:
    result: dict[str, object] = {}
    idx = start
    while idx < len(lines):
        line_indent, content = lines[idx]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indentation at line: {content}")
        if content.startswith("- "):
            break

        key, sep, remainder = content.partition(":")
        if sep == "":
            raise ValueError(f"Invalid mapping entry: {content}")
        key = key.strip()
        remainder = remainder.strip()
        idx += 1
        if remainder == "":
            value, idx = _parse_block(lines, idx, indent + 2)
        else:
            value = _parse_scalar(remainder)
        result[key] = value
    return result, idx


def _parse_list(lines: list[tuple[int, str]], start: int, indent: int) -> tuple[list[object], int]:
    result: list[object] = []
    idx = start
    while idx < len(lines):
        line_indent, content = lines[idx]
        if line_indent < indent:
            break
        if line_indent != indent or not content.startswith("- "):
            break

        item_content = content[2:].strip()
        idx += 1
        if item_content == "":
            value, idx = _parse_block(lines, idx, indent + 2)
            result.append(value)
            continue

        if ":" in item_content:
            key, _, remainder = item_content.partition(":")
            key = key.strip()
            item: dict[str, object] = {key: _parse_scalar(remainder.strip()) if remainder.strip() else None}
            if idx < len(lines) and lines[idx][0] > indent:
                nested, idx = _parse_dict(lines, idx, indent + 2)
                if item[key] is None:
                    item[key] = nested
                else:
                    item.update(nested)
            if item[key] is None:
                item[key] = {}
            result.append(item)
            continue

        result.append(_parse_scalar(item_content))
    return result, idx


def _parse_simple_yaml(text: str) -> dict[str, object]:
    lines = _clean_lines(text)
    if not lines:
        return {}
    parsed, idx = _parse_dict(lines, 0, lines[0][0])
    if idx != len(lines):
        raise ValueError("Failed to parse entire config")
    return parsed


def load_suite_config(config_path: str | Path) -> SuiteConfig:
    path = Path(config_path)
    raw_obj = _parse_simple_yaml(path.read_text(encoding="utf-8"))

    _expect_type(raw_obj, dict, "config")
    raw = cast(dict[str, object], raw_obj)
    suite_name = raw.get("suite_name")
    if not suite_name:
        raise ValueError("suite_name is required")

    seeds_obj = raw.get("seeds", [])
    _expect_type(seeds_obj, list, "seeds")
    seeds_raw = cast(list[object], seeds_obj)

    conditions_obj = raw.get("conditions", [])
    _expect_type(conditions_obj, list, "conditions")
    conditions_raw = cast(list[object], conditions_obj)
    for index, condition in enumerate(conditions_raw):
        _expect_type(condition, dict, f"conditions[{index}]")

    models_obj = raw.get("models", {})
    _expect_type(models_obj, dict, "models")
    models_raw = cast(dict[object, object], models_obj)

    conditions = [_condition_from_mapping(_string_key_dict(cast(dict[object, object], condition))) for condition in conditions_raw]
    models: dict[str, list[ModelConfig]] = {}
    for family, configs in models_raw.items():
        _expect_type(configs, list, f"models.{family}")
        normalized_configs: list[ModelConfig] = []
        configs_list = cast(list[object], configs)
        for index, config in enumerate(configs_list):
            _expect_type(config, dict, f"models.{family}[{index}]")
            normalized_configs.append(ModelConfig(family=str(family), values=_string_key_dict(cast(dict[object, object], config))))
        models[str(family)] = normalized_configs

    output_dir = raw.get("output_dir")
    if output_dir is None:
        output_dir = f"outputs/{suite_name}"

    return SuiteConfig(
        suite_name=str(suite_name),
        output_dir=str(output_dir),
        seeds=[_to_int(seed, "seed") for seed in seeds_raw],
        conditions=conditions,
        models=models,
        trace_plots=_string_key_dict(cast(dict[object, object], raw["trace_plots"])) if "trace_plots" in raw and isinstance(raw["trace_plots"], dict) else None,
        config_path=str(path),
    )
