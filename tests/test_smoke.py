import reinfors


def test_import_reaches_rust_core() -> None:
    v = reinfors._reinfors.core_version()
    assert v and v.count(".") == 2
    assert reinfors.__version__ == v
