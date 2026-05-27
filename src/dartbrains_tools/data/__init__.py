"""DartBrains dataset accessors.

This subpackage exposes one module per dataset (``localizer``, ``sherlock``,
``paranoia``). For back-compat, the Localizer API is also re-exported at the
``dartbrains_tools.data`` namespace level, so existing code that does
``from dartbrains_tools.data import get_subjects`` keeps working.

New code is encouraged to use the per-dataset namespace explicitly::

    from dartbrains_tools.data import sherlock
    bold = sherlock.get_file("sub-01", task="sherlockPart1", suffix="bold")
"""

from . import localizer, paranoia, sherlock
from .localizer import (
    CONDITIONS,
    REPO_ID,
    SUBJECTS,
    TR,
    _download,
    get_file,
    get_subjects,
    get_tr,
    load_confounds,
    load_events,
)

__all__ = [
    "CONDITIONS",
    "REPO_ID",
    "SUBJECTS",
    "TR",
    "_download",
    "get_file",
    "get_subjects",
    "get_tr",
    "load_confounds",
    "load_events",
    "localizer",
    "paranoia",
    "sherlock",
]
