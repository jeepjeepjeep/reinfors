"""reinfors — general-purpose gym-style simulation + batching engine.

The Rust core is exposed as the compiled ``reinfors._reinfors`` extension module. The ergonomic
Python API (the declarative game builder, gym-style vector API) will live in this package as the
project grows; for now it just re-exports the compiled core.
"""

from . import _reinfors

__all__ = ["_reinfors"]
__version__ = "0.0.0"
