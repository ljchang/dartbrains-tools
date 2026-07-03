"""Unit tests for the CLI's pure helpers -- no network."""

import generate_hf_configs as cli


def test_tsv_to_csv_basic():
    tsv = "participant_id\tage\tsex\nS01\t25\tF\nS02\t31\tM\n"
    out = cli._tsv_to_csv(tsv)
    lines = out.strip().splitlines()
    assert lines[0] == "participant_id,age,sex"
    assert lines[1] == "S01,25,F"


def test_tsv_to_csv_quotes_embedded_comma():
    # A field containing a comma must be quoted in the CSV output.
    tsv = "id\tnote\nS01\thello, world\n"
    out = cli._tsv_to_csv(tsv)
    assert '"hello, world"' in out
