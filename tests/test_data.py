"""Shape tests for the localizer dataset module -- does not hit the network."""

from dartbrains_tools.data import localizer


def test_constants():
    assert localizer.REPO_ID == "dartbrains/localizer"
    assert localizer.TR > 0
    assert isinstance(localizer.CONDITIONS, (list, tuple))
    assert len(localizer.CONDITIONS) > 0


def test_get_subjects_returns_list():
    subjects = localizer.get_subjects()
    assert isinstance(subjects, list)
    assert all(isinstance(s, str) for s in subjects)
    assert len(subjects) > 0


def test_get_tr_returns_number():
    assert localizer.get_tr() == localizer.TR


def test_get_file_path_construction(monkeypatch):
    """get_file should construct the right filename without hitting the network."""
    captured = {}

    def fake_download(filename):
        captured["filename"] = filename
        return f"/cache/{filename}"

    monkeypatch.setattr(localizer, "_download", fake_download)

    path = localizer.get_file("S01", scope="derivatives", suffix="bold")
    assert "sub-S01_task-localizer" in captured["filename"]
    assert path.endswith(captured["filename"])
