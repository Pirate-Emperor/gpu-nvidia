"""Static compilation checks for project Python tooling."""
from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parents[1]


def test_python_tooling_compiles():
    paths = [
        *sorted((ROOT / "benchmarks").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        ROOT / "run_attention.py",
    ]
    for path in paths:
        py_compile.compile(str(path), doraise=True)
