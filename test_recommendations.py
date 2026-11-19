"""Checks that the selection logic follows thesis Section 4.4.

Run with:  python -m pytest test_recommendations.py
"""

import pytest

from recommendations import RANKS, TOP, recommend, scope_issues

BASE = {
    "S1": "Yes", "S2": "No", "S3": "No",
    "E1": "Linux", "E2": "Yes", "E3": "Yes", "E4": "More than 32 GB",
    "D1": "Two classes", "D2": "No", "D3": "No", "D4": "No", "D5": "No",
    "D6": "No", "D7": "No", "D8": "No", "D9": "No", "D10": "No", "D11": "No",
    "U1": "No", "U2": "About 15 minutes",
}


def run(**changes):
    return recommend({**BASE, **changes})


# Table 4.10 ------------------------------------------------------------
@pytest.mark.parametrize("question, answer", [("S1", "No"), ("S2", "Yes"), ("S3", "Yes")])
def test_out_of_scope_stops(question, answer):
    answers = {**BASE, question: answer}
    assert scope_issues(answers)
    with pytest.raises(ValueError):
        recommend(answers)


# Table 4.11: Linux, more than 32 GB leaves all tools without warnings ----
def test_unrestricted_environment():
    r = run()
    assert set(r["ranked"]) == set(TOP)
    assert r["alternatives"] == ["LightAutoML", "Lightwood", "TPOT"]
    assert not r["excluded"] and not r["general"]
    assert set(r["notes"]) == {"TPOT", "Lightwood"}   # budget warning, internet note


def test_no_system_software_linux():
    r = run(E3="No", E2="No")
    assert {"H2O AutoML", "auto-sklearn"} <= set(r["excluded"])
    assert "FLAML" not in r["excluded"]


def test_no_system_software_with_containers():
    for system in ("Linux", "macOS"):
        assert not run(E1=system, E3="No", E2="Yes")["excluded"]


def test_no_system_software_macos_without_containers():
    r = run(E1="macOS", E3="No", E2="No")
    assert {"H2O AutoML", "auto-sklearn", "FLAML", "AutoGluon", "LightAutoML"} == set(r["excluded"])
    assert r["ranked"] == ["MLJAR AutoML"]
    assert any("OpenMP" in g for g in r["general"])
    assert not any("OpenMP" in g for g in run(E1="macOS")["general"])


def test_no_containers_macos_and_windows():
    mac = run(E1="macOS", E2="No")
    assert {"auto-sklearn", "LightAutoML"} <= set(mac["excluded"])
    win = run(E1="Windows", E2="No")
    assert "auto-sklearn" in win["excluded"]
    assert "LightAutoML" not in win["excluded"]
    assert any("Windows" in g for g in win["general"])


def test_internet_is_a_note_for_lightwood():
    r = run()
    assert "Lightwood" not in r["excluded"]
    assert any("internet" in n for n in r["notes"]["Lightwood"])


def test_memory_warnings():
    mid = run(E4="16 to 32 GB")
    assert len(mid["notes"]["TPOT"]) == 2 and "AutoGluon" not in mid["notes"]
    low = run(E4="Less than 16 GB")
    assert all(t in low["notes"] for t in ("AutoGluon", "auto-sklearn", "FLAML", "H2O AutoML"))


# Table 4.12 ------------------------------------------------------------
def test_dataset_notes():
    r = run(D2="Yes", D7="Yes", D8="Yes", D9="Yes", D10="Yes", D11="Yes")
    assert "MLJAR AutoML" in r["notes"]
    assert any("target type" in n for n in r["notes"]["Lightwood"])
    assert "H2O AutoML" in r["notes"]  # D8
    assert any("identifiers" in n for n in r["notes"]["Lightwood"])  # D9
    assert "AutoGluon" in r["notes"]
    assert any("empty or constant" in g for g in r["general"])


def test_notes_do_not_change_order():
    assert run(D7="Yes", D8="Yes", D9="Yes", D10="Yes", D11="Yes")["ranked"] == run()["ranked"]


def test_regression_skips_class_questions():
    r = recommend({k: v for k, v in {**BASE, "D1": "A number"}.items()
                   if k not in ("D6", "D7", "D11")})
    assert r["segments"] == ["all"]


def test_imbalance_ignored_for_regression():
    assert run(D1="A number", D6="Yes")["segments"] == ["all"]


def test_multiclass_imbalance_warning():
    assert any("minority class" in g for g in run(D1="More than two classes", D6="Yes")["general"])
    assert not any("minority class" in g for g in run(D6="Yes")["general"])


# Step 4: order (Section 4.4.2) -----------------------------------------
def test_low_segment_everywhere_uses_all_datasets():
    r = run()
    assert r["segments"] == ["all"]
    assert r["ranked"] == ["AutoGluon", "auto-sklearn", "MLJAR AutoML", "FLAML", "H2O AutoML"]


def test_thesis_example():
    """Binary task with more than 100 predictors (Section 4.4.2)."""
    r = run(D3="Yes")
    assert r["ranked"] == ["auto-sklearn", "AutoGluon", "H2O AutoML", "MLJAR AutoML", "FLAML"]
    assert [round(r["scores"][t], 2) for t in r["ranked"]] == [2.57, 3.07, 3.64, 4.00, 4.29]
    assert run(D3="Yes", U1="Yes")["ranked"] == [
        "AutoGluon", "FLAML", "auto-sklearn", "H2O AutoML", "MLJAR AutoML"]


def test_several_segments_are_averaged():
    r = run(D2="Yes", D5="Yes")
    expected = (RANKS["high_sample_size"]["AutoGluon"] + RANKS["high_missing"]["AutoGluon"]) / 2
    assert r["scores"]["AutoGluon"] == pytest.approx(expected)


# Table 4.13 ------------------------------------------------------------
def test_short_budget():
    r = run(U2="Less than 15 minutes")
    assert r["short_budget"] == ["Lightwood", "LightAutoML"]
    assert any("not tested" in g for g in r["general"])
    mac = run(U2="Less than 15 minutes", E1="macOS", E2="No")
    assert mac["short_budget"] == ["Lightwood"]


def test_long_budget():
    assert any("longer" in g for g in run(U2="More than 15 minutes")["general"])


def test_tpot_budget_warning_always():
    assert any("19 of its 29" in n for n in run()["notes"]["TPOT"])


def test_system_software_question_skipped_with_containers():
    answers = {k: v for k, v in BASE.items() if k != "E3"}   # E2 = Yes, E3 not asked
    assert not recommend(answers)["excluded"]


# Corrections after the walkthrough -------------------------------------
def test_reason_names_only_available_tools():
    r = run(E1="macOS", E2="No", E3="No", U1="Yes")
    assert r["ranked"] == ["MLJAR AutoML"]
    reason = r["reasons"]["MLJAR AutoML"]
    assert "AutoGluon" not in reason and "FLAML" not in reason
    linux = run(E2="No", E3="No", U1="Yes")          # H2O and auto-sklearn excluded
    assert linux["ranked"] == ["AutoGluon", "FLAML", "MLJAR AutoML"]
    assert "AutoGluon and FLAML" in linux["reasons"]["MLJAR AutoML"]
    for text in linux["reasons"].values():
        assert "H2O" not in text and "auto-sklearn" not in text


def test_preparation_is_task_specific():
    assert "MLJAR AutoML" in run(U1="Yes")["preparation"]
    for task in ("More than two classes", "A number"):
        r = run(D1=task, U1="Yes")
        assert "MLJAR AutoML" not in r["preparation"]
        assert r["ranked"].index("MLJAR AutoML") < r["ranked"].index("auto-sklearn")
    assert "target" not in run(D1="A number")["preparation"]["H2O AutoML"]


def test_preference_does_not_exclude():
    r = run(U1="Yes")
    assert set(r["ranked"]) == set(TOP)
    assert {"auto-sklearn", "H2O AutoML", "MLJAR AutoML"} <= set(r["preparation"])


def test_not_sure_segments_are_not_used():
    r = run(D2="Not sure", D3="Yes")
    assert r["segments"] == ["high_dimension"]
    assert any("Not sure" in g and "10,000 rows" in g for g in r["general"])
    assert "MLJAR AutoML" in r["notes"]            # D2 note shown as precaution
    assert run(D2="Not sure", D3="Not sure", D4="Not sure", D5="Not sure",
               D6="Not sure")["segments"] == ["all"]


def test_not_sure_notes_shown_as_precaution():
    r = run(D7="Not sure", D8="Not sure", D9="Not sure", D10="Not sure", D11="Not sure")
    assert "H2O AutoML" in r["notes"] and "AutoGluon" in r["notes"]
    assert any("empty or constant" in g for g in r["general"])
    assert r["ranked"] == run()["ranked"]


def test_reasons_are_plain_language():
    r = run(D2="Yes", D4="Yes")
    for text in r["reasons"].values():
        assert "rank" not in text.lower() and "segment" not in text.lower()
    assert "more than 10,000 rows and many categorical predictors" in r["reasons"][r["ranked"][0]]
    assert "same datasets" in r["reasons"][r["ranked"][1]]
