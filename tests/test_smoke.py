import re

import reinfors


def test_import_reaches_rust_core() -> None:
    v = reinfors._reinfors.core_version()
    assert re.match(r"\d+\.\d+\.\d+", v)
    assert reinfors.__version__ == v
