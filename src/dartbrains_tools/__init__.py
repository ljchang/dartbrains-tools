"""DartBrains helper library: data loaders, MR physics simulations, anywidgets."""

__version__ = "0.1.3"

from . import bids, data, mr_simulations, mr_widgets

__all__ = [
    "__version__",
    "bids",
    "data",
    "mr_simulations",
    "mr_widgets",
    "notebook_utils",  # lazy: requires the [notebook] extra (marimo)
]


def __getattr__(name):
    if name == "notebook_utils":
        from . import notebook_utils as _nu
        return _nu
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
