"""DartBrains helper library: data loaders, MR physics simulations, anywidgets."""

__version__ = "0.1.6"

from . import bids, mr_simulations, mr_widgets

__all__ = [
    "__version__",
    "bids",
    "data",            # lazy: pulls in huggingface_hub (see __getattr__)
    "mr_simulations",
    "mr_widgets",
    "notebook_utils",  # lazy: requires the [notebook] extra (marimo)
]


def __getattr__(name):
    # `data` is imported lazily because it transitively imports huggingface_hub,
    # which imports `termios` at load time. `termios` is a Unix-TTY stdlib module
    # absent in Pyodide/WASM, so eager import broke every `mode: wasm` notebook
    # that only needs mr_simulations / mr_widgets. Importing `dartbrains_tools.data`
    # (or `from dartbrains_tools import data`) still works on demand off-WASM.
    #
    # `notebook_utils` is lazy for a different reason: it requires the [notebook]
    # extra (marimo). Both use importlib.import_module -- NOT `from . import name` --
    # because the latter re-enters this __getattr__ and recurses infinitely.
    if name in ("data", "notebook_utils"):
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
