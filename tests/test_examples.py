import importlib
import os
from pathlib import Path
from typing import Any, Optional


def _load_from_file(name: str, sub_path: Optional[str] = None) -> Any:
    path = Path(__file__).parent.parent / "examples"
    if sub_path is not None:
        path = path / sub_path
    path = path / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rosenbrock_deterministic(tmp_path: Path) -> None:
    os.chdir(tmp_path)
    module = _load_from_file("rosenbrock")
    module.main()


def test_rosenbrock_ensemble(tmp_path: Path) -> None:
    os.chdir(tmp_path)
    module = _load_from_file("discrete")
    module.main()
