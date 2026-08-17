import re

import reinfors


def test_import_reaches_rust_core() -> None:
    v = reinfors._reinfors.core_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", v)
    assert reinfors.__version__ == v
