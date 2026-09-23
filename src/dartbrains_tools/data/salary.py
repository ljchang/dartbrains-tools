"""
DartBrains Salary Teaching Data
===============================

The two small tables the pandas, polars and plotting tutorials (and their
assignments) analyze, from HuggingFace Hub (dartbrains/salary):

- ``salary.csv``: professor salaries (``salary, gender, departm, years, age,
  publications``), used throughout the pandas tutorial.
- ``salary_exercise.csv``: Weisberg (1985), *Applied Linear Regression*,
  p. 194 -- 52 tenure-track professors (``sx, rk, yr, dg, yd, sl``).

They used to be read straight from the DartBrains GitHub repository. Going
through HuggingFace like every other course dataset means one way to load data,
a local cache (and, in the browser, marimo's persistent filesystem), and files
the grader can pre-fetch for its offline autograding sandbox.

``get_file`` returns a local path, so notebooks still do the reading themselves::

    from dartbrains_tools.data import salary
    df = pd.read_csv(salary.get_file("salary.csv"))
"""

from typing import Literal

from ._hub import download

REPO_ID = "dartbrains/salary"

FILES = ("salary.csv", "salary_exercise.csv")

File = Literal["salary.csv", "salary_exercise.csv"]


def _download(filename: str) -> str:
    return download(REPO_ID, filename)


def get_file(name: File = "salary.csv") -> str:
    """Download one of the salary tables and return its local path.

    Args:
        name: ``"salary.csv"`` (default) or ``"salary_exercise.csv"``.

    Returns:
        Local path to the cached file.
    """
    if name not in FILES:
        raise ValueError(f"unknown salary file {name!r}; have {', '.join(FILES)}")
    return _download(name)
