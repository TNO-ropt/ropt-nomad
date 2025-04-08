from pathlib import Path

import pytest
from ropt.config.options import gen_options_table

from ropt_nomad.nomad import _OPTIONS_SCHEMA


def test_nomad_options_table() -> None:
    msg = "Regenerate using: python -m ropt_nomad.nomad "
    md_file = Path(__file__).parent.parent / "docs" / "snippets" / "nomad.md"
    if not md_file.exists():
        pytest.fail(f"File not found: {md_file}\n{msg}")
    with md_file.open() as fp:
        saved = fp.read()
    generated = gen_options_table(_OPTIONS_SCHEMA)
    if saved.strip() != generated.strip():
        pytest.fail(
            f"The generated options table does not match the saved version.\n{msg}"
        )
