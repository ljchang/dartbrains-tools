"""Back-compat: the localizer API stays importable at dartbrains_tools.data.*."""

from dartbrains_tools import data


def test_localizer_reexports():
    assert data.REPO_ID == "dartbrains/localizer"
    assert callable(data.get_subjects)
    assert callable(data.get_tr)
    assert callable(data.get_file)
    assert callable(data.load_events)
    assert callable(data.load_confounds)


def test_subpackage_modules_importable():
    from dartbrains_tools.data import localizer, paranoia, sherlock

    assert localizer.REPO_ID == "dartbrains/localizer"
    assert sherlock.REPO_ID == "dartbrains/sherlock"
    assert paranoia.REPO_ID == "dartbrains/paranoia"
