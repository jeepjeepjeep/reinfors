"""Source identity baked into the wheel."""

import re

import reinfors as rf


def test_build_info_identifies_the_source() -> None:
    info = rf.build_info()
    assert set(info) == {"git_sha", "git_dirty", "git_tag", "profile", "version"}
    assert info["version"] == rf.core_version()
    assert info["profile"] in ("debug", "release")
    assert info["git_sha"] == "unknown" or re.fullmatch(r"[0-9a-f]{40}", info["git_sha"])
    assert info["git_dirty"] in (True, False, "unknown")
    assert info["git_tag"] is None or isinstance(info["git_tag"], str)
