"""Shape tests for the salary dataset module -- does not hit the network."""

import pytest

from dartbrains_tools.data import salary


def test_constants():
    assert salary.REPO_ID == "dartbrains/salary"
    assert salary.FILES == ("salary.csv", "salary_exercise.csv")


@pytest.mark.parametrize("name", ["salary.csv", "salary_exercise.csv"])
def test_get_file_downloads_the_named_table(monkeypatch, name):
    captured = {}
    monkeypatch.setattr(salary, "_download", lambda f: captured.setdefault("f", f))
    salary.get_file(name)
    assert captured["f"] == name


def test_get_file_defaults_to_salary_csv(monkeypatch):
    monkeypatch.setattr(salary, "_download", lambda f: f)
    assert salary.get_file() == "salary.csv"


def test_get_file_rejects_unknown_names(monkeypatch):
    monkeypatch.setattr(salary, "_download", lambda f: pytest.fail("should not download"))
    with pytest.raises(ValueError, match="unknown salary file"):
        salary.get_file("salaries.csv")


def test_salary_is_a_public_storage_dataset():
    from dartbrains_tools import storage

    assert storage.dataset("salary").repo == "dartbrains/salary"
