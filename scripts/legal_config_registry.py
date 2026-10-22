#!/usr/bin/env python3
"""Maintain the editable workload/tile registry used by the mentor numerical matrix."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "tests" / "legal_attention_configs.json"
SUPPORTED_D = (32, 64, 128)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer; received {value!r}.")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}; received {value!r}.")
    return value


def validate_config_data(data: Any) -> tuple[list[dict[str, Any]], list[dict[str, int]], list[str]]:
    if not isinstance(data, dict):
        raise ValueError("The JSON root must be an object.")
    if data.get("schema_version") != 1:
        raise ValueError("schema_version must be 1.")
    cases = data.get("cases")
    tiles = data.get("tile_candidates")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list.")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError("tile_candidates must be a non-empty list.")

    clean_cases: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    required_case_fields = ("name", "B", "H", "N", "D", "Dv", "seed")
    for index, item in enumerate(cases):
        where = f"cases[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be a JSON object.")
        missing = [field for field in required_case_fields if field not in item]
        if missing:
            raise ValueError(f"{where} is missing field(s): {', '.join(missing)}.")
        name = item["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", name):
            raise ValueError(
                f"{where}.name must start with a lowercase letter and contain only lowercase "
                "letters, digits and underscores (2–64 characters)."
            )
        if name in seen_names:
            raise ValueError(f"Duplicate case name {name!r}. Case names must be unique.")
        seen_names.add(name)

        case = {"name": name}
        for field in ("B", "H", "N"):
            case[field] = _positive_int(item[field], f"{where}.{field}")
        for field in ("D", "Dv"):
            case[field] = _positive_int(item[field], f"{where}.{field}")
        if case["D"] not in SUPPORTED_D:
            raise ValueError(
                f"{where}.D={case['D']} is unsupported; the kernel supports D=32, 64 or 128."
            )
        if case["Dv"] != case["D"]:
            raise ValueError(
                f"{where}.Dv must equal D for this self-attention implementation "
                f"(received D={case['D']}, Dv={case['Dv']})."
            )
        case["seed"] = _integer(item["seed"], f"{where}.seed")
        clean_cases.append(case)

    clean_tiles: list[dict[str, int]] = []
    seen_tiles: set[tuple[int, int]] = set()
    for index, item in enumerate(tiles):
        where = f"tile_candidates[{index}]"
        if not isinstance(item, dict) or "tile_q" not in item or "tile_k" not in item:
            raise ValueError(f"{where} must contain tile_q and tile_k integers.")
        tq = _positive_int(item["tile_q"], f"{where}.tile_q")
        tk = _positive_int(item["tile_k"], f"{where}.tile_k")
        if not is_power_of_two(tq):
            raise ValueError(f"{where}.tile_q={tq} is not a power of two. Choose 1, 2, 4, 8, 16, 32, 64, 128, and so on.")
        if not is_power_of_two(tk):
            raise ValueError(f"{where}.tile_k={tk} is not a power of two. Choose 1, 2, 4, 8, 16, 32, 64, 128, and so on.")
        pair = (tq, tk)
        if pair in seen_tiles:
            raise ValueError(f"Duplicate tile candidate {tq}x{tk}.")
        seen_tiles.add(pair)
        clean_tiles.append({"tile_q": tq, "tile_k": tk})

    legal_pairs = []
    for case in clean_cases:
        valid_for_case = [
            tile for tile in clean_tiles
            if case["N"] % tile["tile_q"] == 0 and case["N"] % tile["tile_k"] == 0
        ]
        if not valid_for_case:
            raise ValueError(
                f"Case {case['name']} has sequence length N={case['N']}, which is not divisible "
                "by both dimensions of any configured tile. Change N or add a compatible tile."
            )
        for tile in valid_for_case:
            legal_pairs.append((case["name"], tile["tile_q"], tile["tile_k"]))

    warnings: list[str] = []
    for tile in clean_tiles:
        compatible = [
            case["name"] for case in clean_cases
            if case["N"] % tile["tile_q"] == 0 and case["N"] % tile["tile_k"] == 0
        ]
        if not compatible:
            raise ValueError(
                f"Tile {tile['tile_q']}x{tile['tile_k']} is not legal for any configured case. "
                "Choose smaller tiles or add a case with a compatible sequence length."
            )
        skipped = [case["name"] for case in clean_cases if case["name"] not in compatible]
        if skipped:
            warnings.append(
                f"Tile {tile['tile_q']}x{tile['tile_k']} will be skipped for {', '.join(skipped)} "
                "because those sequence lengths are not divisible by both tile dimensions."
            )

    return clean_cases, clean_tiles, warnings


def load_config_data(path: Path = DEFAULT_CONFIG_PATH) -> tuple[list[dict[str, Any]], list[dict[str, int]], list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    return validate_config_data(data)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    cases, tiles, warnings = load_config_data(path)
    legal = [
        (case, tile)
        for case in cases
        for tile in tiles
        if case["N"] % tile["tile_q"] == 0 and case["N"] % tile["tile_k"] == 0
    ]
    return {"cases": cases, "tile_candidates": tiles, "warnings": warnings, "legal_pair_count": len(legal)}


def save_config(path: Path, cases: list[dict[str, Any]], tiles: list[dict[str, int]]) -> None:
    data = {
        "schema_version": 1,
        "description": (
            "Workloads and tile candidates for the extended cuTile numerical-validation matrix. "
            "The required 48-test suite is separate from this editable matrix."
        ),
        "cases": cases,
        "tile_candidates": tiles,
    }
    validate_config_data(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def print_summary(path: Path) -> None:
    config = load_config(path)
    print(f"CONFIGURATION_FILE={path}")
    print(f"WORKLOADS={len(config['cases'])}")
    print(f"TILE_CANDIDATES={len(config['tile_candidates'])}")
    print(f"LEGAL_CONFIGURATIONS={config['legal_pair_count']}")
    print("")
    print(f"{'CASE':28s} {'B':>3s} {'H':>3s} {'N':>6s} {'D':>4s} {'Dv':>4s} {'LEGAL TILES'}")
    for case in config["cases"]:
        legal_tiles = [
            f"{tile['tile_q']}x{tile['tile_k']}"
            for tile in config["tile_candidates"]
            if case["N"] % tile["tile_q"] == 0 and case["N"] % tile["tile_k"] == 0
        ]
        print(
            f"{case['name']:28s} {case['B']:3d} {case['H']:3d} {case['N']:6d} "
            f"{case['D']:4d} {case['Dv']:4d} {', '.join(legal_tiles)}"
        )
    for warning in config["warnings"]:
        print(f"WARNING: {warning}")


def ask_int(label: str, default: int | None = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{label}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print("WARNING: enter a whole number. Try again.")


def prompt_add_case(path: Path) -> None:
    while True:
        cases, tiles, _ = load_config_data(path)
        print("Add workload dimensions. A supported head dimension is 32, 64 or 128.")
        name = input("Unique workload name (lowercase letters, digits, underscores): ").strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", name):
            print("WARNING: use 2–64 lowercase letters/digits/underscores and start with a letter. Try again.")
            continue
        if any(case["name"] == name for case in cases):
            print(f"WARNING: {name!r} already exists. Choose another name.")
            continue

        b = ask_int("Batch size B", 1)
        h = ask_int("Attention heads H", 8)
        n = ask_int("Sequence length N", 512)
        d = ask_int("Head dimension D (32/64/128)", 128)
        seed = ask_int("Random seed", max(case["seed"] for case in cases) + 1)
        try:
            for label, value in (("B", b), ("H", h), ("N", n)):
                _positive_int(value, label)
            if d not in SUPPORTED_D:
                raise ValueError(f"D must be one of {SUPPORTED_D}; received {d}.")
            case = {"name": name, "B": b, "H": h, "N": n, "D": d, "Dv": d, "seed": seed}
            compatible = [
                (tile["tile_q"], tile["tile_k"]) for tile in tiles
                if n % tile["tile_q"] == 0 and n % tile["tile_k"] == 0
            ]
            if not compatible:
                print(
                    f"WARNING: N={n} is not divisible by both dimensions of any configured tile. "
                    "Change N or first add a compatible tile. Nothing was saved."
                )
                continue
            work = b * h * n * n
            if work > 134_217_728:
                print(
                    "WARNING: B*H*N*N exceeds 134 million score positions. The reference calculation "
                    "can use substantial CPU RAM and GPU work. Consider a shorter sequence or smaller B/H."
                )
                confirm = input("Continue with this large test case? Type YES to continue: ").strip()
                if confirm != "YES":
                    print("Entry cancelled. Try again with smaller dimensions.")
                    continue
            cases.append(case)
            save_config(path, cases, tiles)
            new_count = sum(
                1 for current in cases for tile in tiles
                if current["N"] % tile["tile_q"] == 0 and current["N"] % tile["tile_k"] == 0
            )
            print(f"ADDED_CASE={name}")
            print(f"NEW_LEGAL_CONFIGURATION_COUNT={new_count}")
            print_summary(path)
            return
        except ValueError as exc:
            print(f"WARNING: {exc} Try the entry again.")


def prompt_add_tile(path: Path) -> None:
    while True:
        cases, tiles, _ = load_config_data(path)
        print("Tile dimensions must be positive powers of two. A tile is included for a case only when both tile dimensions divide N.")
        tq = ask_int("Query tile TQ", 64)
        tk = ask_int("Key/value tile TK", 64)
        try:
            _positive_int(tq, "TQ")
            _positive_int(tk, "TK")
            if not is_power_of_two(tq):
                raise ValueError(f"TQ={tq} is not a power of two.")
            if not is_power_of_two(tk):
                raise ValueError(f"TK={tk} is not a power of two.")
            if any(tile["tile_q"] == tq and tile["tile_k"] == tk for tile in tiles):
                raise ValueError(f"Tile {tq}x{tk} already exists.")
            compatible = [
                case["name"] for case in cases
                if case["N"] % tq == 0 and case["N"] % tk == 0
            ]
            if not compatible:
                raise ValueError(
                    f"Tile {tq}x{tk} is not divisible into any configured sequence length. "
                    "Try a smaller tile or add a compatible workload first."
                )
            tiles.append({"tile_q": tq, "tile_k": tk})
            save_config(path, cases, tiles)
            skipped = [case["name"] for case in cases if case["name"] not in compatible]
            print(f"ADDED_TILE={tq}x{tk}; legal for: {', '.join(compatible)}")
            if skipped:
                print(f"WARNING: this tile will be excluded for: {', '.join(skipped)}")
            print_summary(path)
            return
        except ValueError as exc:
            print(f"WARNING: {exc} Try the entry again.")


def interactive(path: Path) -> None:
    while True:
        print("")
        print("=" * 72)
        print("LEGAL NUMERICAL CONFIGURATION MENU")
        print(f"File: {path}")
        print("1. Add a workload (B, H, N, D)")
        print("2. Add a tile candidate (TQ, TK)")
        print("3. List current workloads and legal combinations")
        print("4. Validate JSON file without running the GPU")
        print("0. Return to the main mentor menu")
        choice = input("Select: ").strip()
        try:
            if choice == "1":
                prompt_add_case(path)
            elif choice == "2":
                prompt_add_tile(path)
            elif choice == "3":
                print_summary(path)
            elif choice == "4":
                cases, tiles, warnings = load_config_data(path)
                print(f"CONFIG_VALID=PASS; WORKLOADS={len(cases)}; TILES={len(tiles)}; LEGAL_CONFIGURATIONS={load_config(path)['legal_pair_count']}")
                for warning in warnings:
                    print(f"WARNING: {warning}")
            elif choice == "0":
                return
            else:
                print("WARNING: choose one of the listed menu numbers.")
        except ValueError as exc:
            print(f"CONFIGURATION ERROR: {exc}")
            print("No invalid configuration has been saved.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("interactive", "list", "validate"), nargs="?", default="interactive")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    try:
        if args.action == "interactive":
            interactive(args.config)
        elif args.action == "list":
            print_summary(args.config)
        else:
            config = load_config(args.config)
            print(
                f"CONFIG_VALID=PASS; WORKLOADS={len(config['cases'])}; "
                f"TILES={len(config['tile_candidates'])}; LEGAL_CONFIGURATIONS={config['legal_pair_count']}"
            )
            for warning in config["warnings"]:
                print(f"WARNING: {warning}")
        return 0
    except (ValueError, OSError) as exc:
        print(f"CONFIG_VALID=FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
