"""Selection logic of the framework (thesis Section 4.4.2).

The rules follow Tables 4.10 to 4.13. The average ranks come from Table 4.8
(data/mean_ranks.csv), and the top group from the overall Friedman and
Nemenyi comparison (Section 4.3.5). This module contains no user interface
code, so the rules can be tested on their own (test_recommendations.py).
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

DATA = Path(__file__).parent / "data"

# Top group of the overall comparison, in the order of the overall average
# ranks (Table 4.8). Ties in the ordering keep this order.
TOP = ("AutoGluon", "auto-sklearn", "MLJAR AutoML", "FLAML", "H2O AutoML")
OTHER = ("LightAutoML", "Lightwood", "TPOT")
# Tools highlighted for a training time of less than 15 minutes (U2).
SHORT_BUDGET = ("Lightwood", "LightAutoML")

# Dataset questions D2 to D6: (segment key in mean_ranks.csv, label).
SEGMENTS = {
    "D2": ("high_sample_size", "More than 10,000 rows"),
    "D3": ("high_dimension", "More than 100 predictors"),
    "D4": ("high_categorical", "More than one third categorical predictors"),
    "D5": ("high_missing", "More than one third of rows with missing values"),
    "D6": ("high_imbalance", "Most frequent class more than ten times as frequent as the least frequent"),
}
SEGMENT_LABELS = {"all": "All benchmark datasets"}
SEGMENT_LABELS.update(dict(SEGMENTS.values()))

# Short descriptions of the high segments for the plain-language reasons.
SEGMENT_PHRASES = {
    "high_sample_size": "more than 10,000 rows",
    "high_dimension": "more than 100 predictors",
    "high_categorical": "many categorical predictors",
    "high_missing": "many rows with missing values",
    "high_imbalance": "strongly imbalanced classes",
}

TASK_TYPES = ("Two classes", "More than two classes", "A number")
NOT_SURE = "Not sure"
MEMORY = ("Less than 16 GB", "16 to 32 GB", "More than 32 GB")
TIME = ("Less than 15 minutes", "About 15 minutes", "More than 15 minutes")

with (DATA / "mean_ranks.csv").open(newline="", encoding="utf-8") as file:
    RANKS = {
        row["analysis"]: {tool: float(row[tool]) for tool in TOP + OTHER}
        for row in csv.DictReader(file)
    }


def is_classification(a):
    return a.get("D1") in TASK_TYPES[:2]


def scope_issues(a):
    """Step 1: scope questions S1 to S3 (Table 4.10)."""
    return [
        explanation for applies, explanation in (
            (a.get("S1") == "No",
             "The framework requires a single table that contains the column to predict."),
            (a.get("S2") == "Yes",
             "The framework does not cover free text or forecasting tasks."),
            (a.get("S3") == "Yes",
             "The benchmark covered at most 100,000 rows and 1,000 predictor columns."),
        ) if applies
    ]


def applies(a, question):
    """A note or warning is shown for Yes and, as a precaution, for Not sure."""
    return a.get(question) in ("Yes", NOT_SURE)


def unanswered_segments(a):
    """Labels of the characteristics D2 to D6 answered with Not sure."""
    classification = is_classification(a)
    return [label for question, (_, label) in SEGMENTS.items()
            if (classification or question != "D6") and a.get(question) == NOT_SURE]


def preparation(tool, a):
    """Preparation step the tool needed in the tested setup, or None."""
    if tool == "auto-sklearn":
        return ("Convert categorical columns to the pandas `category` type, otherwise text "
                "columns are treated as free text.")
    if tool == "H2O AutoML":
        target = " and the target" if is_classification(a) else ""
        return f"Declare categorical columns{target} as categorical (`enum`)."
    if tool == "MLJAR AutoML" and a.get("D1") == TASK_TYPES[0]:
        return ("Encode the two class labels as 0 and 1 and map the predictions back to the "
                "original labels.")
    return None


def high_segments(a):
    """Segment keys for which the dataset lies in the high segment.

    D6 (class imbalance) applies to classification only. Not sure counts as
    not high. If no high segment applies, the ranks over all 30 datasets are
    used.
    """
    classification = is_classification(a)
    keys = [
        key for question, (key, _) in SEGMENTS.items()
        if (classification or question != "D6") and a.get(question) == "Yes"
    ]
    return keys or ["all"]


def recommend(a):
    if scope_issues(a):
        raise ValueError("The task is outside the framework scope.")

    exclusions = defaultdict(list)
    notes = defaultdict(list)
    general = []
    classification = is_classification(a)
    containers = a["E2"] == "Yes"

    # Step 2: environment (Table 4.11). Containers (E2) are asked first,
    # because a container provides Java, OpenMP and build tools itself. The
    # system-software question (E3) is only asked if no container can be
    # used, and tools that need system software are excluded only then.
    if not containers and a.get("E3") == "No":
        neither = "and neither system software nor containers can be used."
        exclusions["H2O AutoML"].append(f"Requires a Java runtime, {neither}")
        exclusions["auto-sklearn"].append(f"Requires build tools, {neither}")
        if a["E1"] == "macOS":
            exclusions["FLAML"].append(
                f"Failed on macOS without the OpenMP library, {neither}")
            exclusions["AutoGluon"].append(
                f"Requires the OpenMP library on macOS according to its documentation, {neither}")
            general.append(
                "Other tools that use LightGBM or XGBoost may also fail on macOS without the "
                "OpenMP library. Only FLAML was tested without it.")
    if a["E1"] in ("macOS", "Windows") and not containers:
        exclusions["auto-sklearn"].append(
            f"Officially supports Linux only, so on {a['E1']} it needs a Linux container.")
    if a["E1"] == "macOS" and not containers:
        exclusions["LightAutoML"].append(
            "Its training did not finish on macOS in the tests and ran only in a Linux container.")
    # Internet access is not asked: it concerns only Lightwood, which is
    # outside the top group, so it is covered by a note (Section 4.4.1).
    notes["Lightwood"].append(
        "Needs internet access during training to download its NLTK language data, "
        "unless the data are installed beforehand.")
    if a["E1"] == "Windows":
        general.append("The tools were not tested on Windows.")
    if a["E4"] != MEMORY[2]:
        notes["TPOT"].append(
            "Reached memory peaks of 24 to 30 GB on three datasets, including one with neither "
            "many rows nor many predictors, and its processes ran out of memory on KDDCup09 appetency.")
    if a["E4"] == MEMORY[0]:
        # Peak memory per tool (Table 4.7), reached on datasets with more
        # than 100 predictors.
        for tool, peak in (("AutoGluon", 14), ("auto-sklearn", 13),
                           ("FLAML", 15), ("H2O AutoML", 15)):
            notes[tool].append(
                f"Reached a memory peak of about {peak} GB on a dataset with more than "
                "100 predictors.")

    # Dataset questions (Table 4.12): notes and warnings, no change of order.
    unknown = unanswered_segments(a)
    if unknown:
        general.append(
            "You answered Not sure for: " + "; ".join(u[0].lower() + u[1:] for u in unknown)
            + ". These characteristics were not used for the order. The Python snippet in the "
            "dataset section shows how to check them.")
    if applies(a, "D2"):
        notes["MLJAR AutoML"].append(
            "Ranked significantly lower than AutoGluon on the nine datasets with more than 10,000 rows.")
    if a["D1"] == TASK_TYPES[1] and applies(a, "D6"):
        general.append(
            "On the two strongly imbalanced multiclass datasets, every tool missed at least one "
            "minority class completely. Check per-class recall and F1, not only an overall score.")
    if classification and applies(a, "D7"):
        notes["Lightwood"].append(
            "Declare the target type: Lightwood treated class labels stored as numbers as a numeric target.")
    if applies(a, "D8"):
        notes["H2O AutoML"].append(
            "Declare the column types: H2O AutoML chooses the column types itself unless they are "
            "declared, so categories stored as numeric codes may be read as numbers.")
    if applies(a, "D9"):
        notes["Lightwood"].append(
            "Removed columns that it classified as identifiers. Check which predictors were retained.")
    if applies(a, "D10"):
        general.append(
            "Remove empty or constant columns before training. TPOT failed on empty columns, while "
            "Lightwood and H2O AutoML removed such columns without stopping the run.")
    if classification and applies(a, "D11"):
        notes["AutoGluon"].append(
            "Removed classes with fewer than ten examples in the training data before training in "
            "the tested setup, so these classes were never predicted.")

    # Steps 3 and 4: top group, ordered by the average ranks of the high
    # segments (Table 4.8), averaged over the segments.
    segments = high_segments(a)
    scores = {tool: mean(RANKS[key][tool] for key in segments) for tool in TOP + OTHER}
    # U1 is a preference, not a requirement: tools that need a preparation
    # step stay recommended but are placed after those that need none.
    prefer_low_prep = a["U1"] == "Yes"
    prep = {tool: preparation(tool, a) for tool in TOP + OTHER}
    ranked = sorted(
        (tool for tool in TOP if tool not in exclusions),
        key=lambda tool: (prefer_low_prep and prep[tool] is not None, scores[tool]),
    )

    # Step 5: tools outside the top group (Table 4.13).
    alternatives = [tool for tool in OTHER if tool not in exclusions]
    short = a["U2"] == TIME[0]
    short_budget = [tool for tool in SHORT_BUDGET if short and tool in alternatives]
    # The budget qualification is returned separately, so that the output can
    # show it next to the starting recommendation.
    budget_note = None
    if short:
        budget_note = ("Budgets shorter than 15 minutes were not tested, so the framework makes "
                       "no statement about which tool performs best with less time.")
    if a["U2"] == TIME[2]:
        budget_note = "Budgets longer than 15 minutes were not tested."
    notes["TPOT"].append(
        "Exceeded the budget by more than the tolerance of 10% in 19 of its 29 completed runs, "
        "by up to 65%.")

    return {
        "ranked": ranked,
        "alternatives": alternatives,
        "short_budget": short_budget,
        "excluded": dict(exclusions),
        "notes": dict(notes),
        "general": general,
        "budget_note": budget_note,
        "scores": scores,
        "segments": segments,
        "prefer_low_prep": prefer_low_prep,
        "preparation": {tool: step for tool, step in prep.items() if step},
        "reasons": {tool: reason(tool, ranked, prep, segments, prefer_low_prep)
                    for tool in ranked},
    }


ORDINALS = ("best", "second-best", "third-best", "fourth-best", "fifth-best")


def _join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def reason(tool, ranked, prep, segments, prefer_low_prep):
    """Plain-language reason for the position of a recommended tool.

    Only tools that remain available in the selected environment are named.
    """
    if segments == ["all"]:
        basis = "across all 30 benchmark datasets"
    else:
        basis = ("on the benchmark datasets that, like yours, have "
                 + _join([SEGMENT_PHRASES[key] for key in segments]))
    if len(ranked) == 1:
        return "The only recommended tool that is available in your environment."
    # The datasets are named in the first reason only, to avoid repetition.
    if ranked.index(tool) > 0:
        basis = "on the same datasets"
    no_prep = [t for t in ranked if prep[t] is None]
    needs_prep = [t for t in ranked if prep[t] is not None]
    if prefer_low_prep and no_prep and needs_prep:
        if prep[tool] is None:
            return (f"Accepts the table without column-type declarations or class-label "
                    f"recoding in the tested setup, as you preferred. Among such tools, it had "
                    f"the {ORDINALS[no_prep.index(tool)]} average result {basis}.")
        verb = "accepts" if len(no_prep) == 1 else "accept"
        return (f"Placed after {_join(no_prep)}, which {verb} the table without these steps. "
                f"Among the tools that need a type declaration or label recoding, it had the "
                f"{ORDINALS[needs_prep.index(tool)]} average result {basis}.")
    return (f"The {ORDINALS[ranked.index(tool)]} average result among the available "
            f"recommended tools {basis}.")
