from __future__ import annotations

import reinfors as rf


def test_catalogue_names_match_runtime_registries() -> None:
    assert set(rf.games.registered()) == set(rf.catalog.GAMES)
    assert set(rf.policies.registered()) == set(rf.catalog.POLICIES)
    assert set(rf.learners.registered()) == set(rf.catalog.LEARNERS)
    assert set(rf.encoders.registered()) == set(rf.catalog.ENCODERS)
    assert set(rf.chance_modes.registered()) == set(rf.catalog.CHANCE_MODES)
    assert set(rf.noise.registered()) == set(rf.catalog.NOISE)
