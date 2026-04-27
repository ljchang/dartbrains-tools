"""Shape tests for the data module — does not hit the network."""

from dartbrains_tools import data


def test_constants():
    assert data.REPO_ID == "dartbrains/localizer"
    assert data.TR > 0
    assert isinstance(data.CONDITIONS, (list, tuple))
    assert len(data.CONDITIONS) > 0


def test_get_subjects_returns_list():
    subjects = data.get_subjects()
    assert isinstance(subjects, list)
    assert all(isinstance(s, str) for s in subjects)
    assert len(subjects) > 0


def test_get_tr_returns_number():
    assert data.get_tr() == data.TR
