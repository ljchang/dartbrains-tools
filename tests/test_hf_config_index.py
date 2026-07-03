"""Pure index-building + README-rendering tests -- no network."""

from hf_configs.index import (
    build_index,
    glob_to_regex,
    render_readme_configs,
    replace_configs_block,
    rows_to_csv,
)
from hf_configs.labels import extract_beta_labels, parse_bids_entities

FILES = [
    "derivatives/fmriprep/sub-S01/func/sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "derivatives/fmriprep/sub-S02/func/sub-S02_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "derivatives/fmriprep/sub-S01/func/sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    "derivatives/betas/S01_beta_audio_computation.nii.gz",
    "derivatives/betas/S01_betas.nii.gz",
    "derivatives/betas/metadata.csv",           # must be excluded by the nifti glob
    "sub-S01/func/sub-S01_task-localizer_events.tsv",
]


def test_glob_star_stops_at_slash():
    rx = glob_to_regex("derivatives/fmriprep/*/func/*_bold.nii.gz")
    # single * does NOT cross a directory boundary
    assert not rx.match(
        "derivatives/fmriprep/sub-S01/anat/extra/sub-S01_desc-preproc_bold.nii.gz"
    )


def test_glob_doublestar_crosses_slash():
    rx = glob_to_regex("derivatives/fmriprep/**/*_desc-preproc_bold.nii.gz")
    assert rx.match(FILES[0])


def test_build_index_bold_uses_bids_labels():
    cfg = {"glob": "derivatives/fmriprep/**/*_desc-preproc_bold.nii.gz"}
    rows = build_index(FILES, cfg)
    assert [r["path"] for r in rows] == [FILES[0], FILES[1]]
    assert rows[0]["subject"] == "S01"
    assert rows[0]["task"] == "localizer"


def test_build_index_betas_excludes_metadata_csv():
    cfg = {"glob": "derivatives/betas/*.nii.gz", "labels": extract_beta_labels}
    rows = build_index(FILES, cfg)
    paths = [r["path"] for r in rows]
    assert "derivatives/betas/metadata.csv" not in paths
    assert {r["type"] for r in rows} == {"individual", "stacked"}


def test_rows_to_csv_header_and_blanks():
    rows = [
        {"path": "a.nii.gz", "subject": "S01", "type": "individual", "condition": "x"},
        {"path": "b.nii.gz", "subject": "S01", "type": "stacked"},
    ]
    csv = rows_to_csv(rows)
    lines = csv.strip().splitlines()
    assert lines[0] == "path,subject,type,condition"
    assert lines[2] == "b.nii.gz,S01,stacked,"   # missing condition -> empty cell


def test_render_readme_configs_marks_default_and_points_at_csv():
    dataset = {
        "repo": "dartbrains/localizer",
        "default": "betas",
        "configs": {
            "bold": {"glob": "x/**"},
            "betas": {"glob": "y/*.nii.gz", "labels": extract_beta_labels},
        },
    }
    yaml = render_readme_configs(dataset)
    assert "config_name: bold" in yaml
    assert "path: bold.csv" in yaml
    assert "config_name: betas" in yaml
    assert "default: true" in yaml
    # exactly one default
    assert yaml.count("default: true") == 1


def test_specs_cover_expected_configs():
    from hf_configs.specs import DATASETS

    assert set(DATASETS) == {
        "dartbrains/localizer",
        "dartbrains/sherlock",
        "dartbrains/paranoia",
    }
    loc = DATASETS["dartbrains/localizer"]["configs"]
    assert set(loc) == {"bold", "confounds", "mask", "events", "betas", "participants"}
    assert set(DATASETS["dartbrains/sherlock"]["configs"]) == {
        "bold", "confounds", "mask", "onsets",
    }
    assert set(DATASETS["dartbrains/paranoia"]["configs"]) == {
        "bold", "confounds", "mask", "participants",
    }
    # participants is a content config; bold is an index config
    assert "content" in loc["participants"]
    assert "glob" in loc["bold"]


def test_all_three_share_the_fmriprep_globs():
    from hf_configs.specs import DATASETS

    globs = {
        repo: DATASETS[repo]["configs"]["bold"]["glob"]
        for repo in DATASETS
    }
    assert len(set(globs.values())) == 1  # identical bold glob everywhere


def test_replace_configs_block_tolerates_crlf():
    from hf_configs.index import replace_configs_block

    readme = "---\r\nlicense: x\r\nconfigs:\r\n  - config_name: old\r\n---\r\n# Body\r\n"
    out = replace_configs_block(readme, "configs:\n  - config_name: new\n")
    assert "license: x" in out
    assert "config_name: new" in out
    assert "config_name: old" not in out


def test_replace_configs_block_preserves_sibling_keys():
    from hf_configs.index import replace_configs_block

    readme = (
        "---\n"
        "license: cc-by-nc-4.0\n"
        "configs:\n"
        "  - config_name: old\n"
        "    data_files:\n"
        "      - split: train\n"
        "        path: old.csv\n"
        "pretty_name: Foo\n"
        "size_categories:\n"
        "  - 1K<n<10K\n"
        "---\n"
        "# Body\n"
    )
    out = replace_configs_block(readme, "configs:\n  - config_name: new\n")
    assert "pretty_name: Foo" in out          # sibling AFTER configs survives
    assert "size_categories" in out
    assert "license: cc-by-nc-4.0" in out      # sibling BEFORE configs survives
    assert "config_name: old" not in out       # old block gone
    assert "config_name: new" in out           # new block present
    assert out.endswith("# Body\n")            # body untouched
