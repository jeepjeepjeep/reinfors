"""reinfors — general-purpose gym-style simulation + batching engine.

The Rust core is exposed as the compiled ``reinfors._reinfors`` extension module. The ergonomic
Python API (the declarative game builder, gym-style vector API) will live in this package as the
project grows; for now it re-exports the compiled core and the game registry.
"""

from . import _reinfors, games

__all__ = ["_reinfors", "games"]
__version__ = "0.0.0"
