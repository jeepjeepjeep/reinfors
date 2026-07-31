"""Checks that maintained documentation examples do not drift from runnable files."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_training_guide_matches_runnable_example() -> None:
    guide = (ROOT / "docs/guides/training.md").read_text()
    documented = guide.split("```python\n", 1)[1].split("\n```", 1)[0] + "\n"
    runnable = (ROOT / "examples/train_gridworld.py").read_text()
    assert documented == runnable
