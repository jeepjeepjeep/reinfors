import reinfors


def test_import_reaches_rust_core() -> None:
    assert reinfors._reinfors.core_version() == "0.0.0"
