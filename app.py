"""Streamlit interface of the AutoML Selection Guide (thesis Section 4.4).

Run with:  streamlit run app.py
The questionnaire follows Tables 4.10 to 4.13, the rules are in
recommendations.py, and the tool cards follow Appendix D.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from recommendations import MEMORY, RANKS, SEGMENT_LABELS, TASK_TYPES, TIME, recommend

DATA = Path(__file__).parent / "data"
CARDS = json.loads((DATA / "tool_cards.json").read_text(encoding="utf-8"))
DOCS = {
    "AutoGluon": "https://auto.gluon.ai/",
    "auto-sklearn": "https://automl.github.io/auto-sklearn/",
    "FLAML": "https://microsoft.github.io/FLAML/",
    "H2O AutoML": "https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html",
    "LightAutoML": "https://lightautoml.readthedocs.io/",
    "Lightwood": "https://mindsdb.github.io/lightwood/",
    "MLJAR AutoML": "https://supervised.mljar.com/",
    "TPOT": "https://epistasislab.github.io/tpot/",
}
# Observed fit times under the 15-minute budget (Table 4.7), for U2.
FIT_TIMES = {
    "Lightwood": "median 17 seconds, maximum about 6 minutes",
    "LightAutoML": "median 176 seconds, maximum about 9 minutes",
}
YES_NO = ("Yes", "No")
CHECK_SIZE = """import pandas as pd

df = pd.read_csv("your_table.csv")   # your table
target = "column_to_predict"         # the column to predict
X, y = df.drop(columns=[target]), df[target]

print("Rows:", len(df))
print("Predictor columns:", X.shape[1])
categorical = X.select_dtypes(exclude="number").columns
print("Share of categorical predictors:",
      round(len(categorical) / X.shape[1], 2))
print("Share of rows with a missing value:",
      round(X.isna().any(axis=1).mean(), 2))

# Classification only
counts = y.value_counts()
print("Most / least frequent class:",
      round(counts.max() / counts.min(), 1))
"""
CHECK_FORMAT = """# Continue with df, X, and y from the first script
print("Empty or constant columns:",
      [c for c in X if X[c].nunique(dropna=True) <= 1])

# Classification only
print("Labels stored as numbers:",
      pd.api.types.is_numeric_dtype(y))
print("Smallest class:", y.value_counts().min())
"""

st.set_page_config(page_title="AutoML Selection Guide", layout="centered")
st.title("AutoML Selection Guide", anchor=False)
st.write(
    "Find open-source AutoML tools to try for a supervised learning task on tabular data. "
    "Answer questions about the task, your environment, your dataset, and your preferences."
)
st.caption(
    "Based on the master thesis A Framework for Using AutoML (Tatiana Korzineva, FHNW, 2026). "
    "You do not upload any data. The recommendation uses only your answers."
)
guide, library = st.tabs(["Get recommendations", "Browse tool cards"])


def card_text(value, task):
    """Card field as text. Classification-specific parts are labelled, and
    hidden for regression (task "regression") in the results."""
    if isinstance(value, str):
        return value
    general, specific = value["general"], value["classification"]
    if task == "regression":
        return general or "None for regression tasks."
    if task == "classification":
        return " ".join(part for part in (general, specific) if part)
    labelled = f"*For classification tasks:* {specific}"
    return f"{general}  \n{labelled}" if general else labelled


def show_card(tool, task=None):
    for label, value in CARDS[tool].items():
        st.markdown(f"**{label}**  \n{card_text(value, task)}")
    st.markdown(f"[Open official documentation]({DOCS[tool]})")


with library:
    st.caption(
        "These cards describe the versions and settings tested in the thesis. Resource use was "
        "observed with a budget of 15 minutes, 8 CPU cores, and 32 GB of memory. It is not a "
        "minimum requirement or a guaranteed upper limit. Passages marked for classification "
        "tasks do not apply to regression."
    )
    for tool in CARDS:
        with st.expander(tool):
            show_card(tool)


answers = {}
# Question texts by key, in the order shown, to name unanswered questions.
LABELS = {}
# Keep all answers when questions are hidden, for example when the scope box
# is unticked and ticked again. Streamlit otherwise removes the state of
# widgets that are not shown in a run.
for _key in list(st.session_state):
    if _key != "submitted_answers":
        st.session_state[_key] = st.session_state[_key]


def ask(key, label, options=YES_NO, help_text=None):
    LABELS[key] = label
    answers[key] = st.radio(label, options, index=None, key=key, help=help_text,
                            horizontal=len(options) <= 3)


def show_preparation(tool, result):
    step = result["preparation"].get(tool)
    if not step:
        return
    if result["prefer_low_prep"]:
        st.warning(f"Needs a column-type declaration or label recoding, which you preferred to "
                   f"avoid: {step} Your answer is treated as a preference, so the tool is still "
                   "recommended.")
    else:
        st.markdown(f"Required in the tested setup: {step}")


BASIS = ("Based on the benchmark of eight AutoML tools carried out in October 2026, with the "
         "tool versions named in the tool cards. Newer versions may behave differently.")


def build_report(answers, result):
    """Plain-text record of the answers and the result, for download."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = ["AutoML Selection Guide: answers and recommendations",
             f"Created: {now}",
             BASIS,
             "",
             "YOUR ANSWERS",
             "- The task meets the scope conditions: Yes"]
    lines += [f"- {LABELS[key]} {value}" for key, value in answers.items() if key in LABELS]

    lines += ["", "RECOMMENDED TOOLS (in this order)"]
    if not result["ranked"]:
        lines.append("None. Your environment excludes all five recommended tools.")
    for position, tool in enumerate(result["ranked"], 1):
        lines.append(f"{position}. {tool}")
        lines.append(f"   Reason: {result['reasons'][tool]}")
        step = result["preparation"].get(tool)
        if step:
            lines.append(f"   Preparation: {step}")
        lines += [f"   Note: {note}" for note in result["notes"].get(tool, [])]
    if result["budget_note"]:
        lines.append(result["budget_note"])

    if result["general"]:
        lines += ["", "BEFORE YOU START"]
        lines += [f"- {message}" for message in result["general"]]

    if result["alternatives"]:
        lines += ["", "OTHER AVAILABLE TOOLS (significantly weaker results across all 30 "
                      "benchmark datasets)"]
        for tool in result["alternatives"]:
            lines.append(f"- {tool}")
            step = result["preparation"].get(tool)
            if step:
                lines.append(f"   Preparation: {step}")
            lines += [f"   Note: {note}" for note in result["notes"].get(tool, [])]

    if result["excluded"]:
        lines += ["", "TOOLS EXCLUDED BY YOUR ENVIRONMENT"]
        for tool, reasons in result["excluded"].items():
            lines.append(f"- {tool}: " + " ".join(reasons))

    lines += ["", "App: https://automl-selection-guide.streamlit.app"]
    return "\n".join(lines) + "\n"


def show_results(result, task, answers):
    st.header("Your recommendations", anchor=False)
    ranked = result["ranked"]
    budget = f" {result['budget_note']}" if result["budget_note"] else ""
    if len(ranked) == 1:
        st.success(f"Start with {ranked[0]}, the only recommended tool available in your "
                   f"environment.{budget}")
    elif ranked:
        st.success(f"Start with {ranked[0]}. If it does not suit your data or setup, try "
                   f"{ranked[1]} next.{budget}")
    elif budget:
        st.warning(budget.strip())
    st.download_button(
        "Download answers and recommendations",
        data=build_report(answers, result),
        file_name=f"automl_recommendations_{datetime.now(timezone.utc):%Y%m%d_%H%M}.txt",
        mime="text/plain",
        help="Saves your answers and the full result as a text file.",
    )

    st.subheader("Recommended tools", anchor=False)
    st.caption(
        "Five tools formed the leading benchmark group. Those available for your environment "
        "are listed below. Across all 30 benchmark datasets, the differences between them were "
        "too small to be statistically "
        "significant, so each of them is a reasonable choice. The order is a tie-breaker based on benchmark datasets that "
        "resemble yours, not a prediction of the best tool for your data, so treat the list "
        "as a shortlist to try on your own data."
    )
    if not result["ranked"]:
        st.info("Your environment excludes all five recommended tools. "
                "Review the other available tools below.")
    elif result["prefer_low_prep"] and all(t in result["preparation"] for t in result["ranked"]):
        st.info("None of the recommended tools that are available in your environment accepts "
                "the table without column-type declarations or class-label recoding, so the "
                "order follows the benchmark results only.")
    for position, tool in enumerate(result["ranked"], 1):
        with st.container(border=True):
            st.markdown(f"**{position}. {tool}**")
            st.write(result["reasons"][tool])
            show_preparation(tool, result)
            for note in result["notes"].get(tool, []):
                st.markdown(f"- {note}")
            with st.expander(f"View {tool} tool card"):
                show_card(tool, task)

    if result["general"]:
        st.subheader("Before you start", anchor=False)
        st.warning("\n".join(f"- {message}" for message in result["general"]))

    if result["alternatives"]:
        st.subheader("Other available tools", anchor=False)
        for tool in result["alternatives"]:
            highlighted = tool in result["short_budget"]
            label = f"{tool} (possible alternative for less than 15 minutes)" if highlighted else tool
            with st.expander(label, expanded=highlighted):
                st.write("Achieved significantly weaker results than the best tool across all 30 "
                         "benchmark datasets.")
                if highlighted:
                    st.write(f"Finished well before the 15-minute budget in the benchmark "
                             f"({FIT_TIMES[tool]}). These times do not guarantee a similar "
                             "time on a new dataset.")
                show_preparation(tool, result)
                for note in result["notes"].get(tool, []):
                    st.markdown(f"- {note}")
                show_card(tool, task)

    if result["excluded"]:
        with st.expander("Tools excluded by your environment", expanded=True):
            for tool, reasons in result["excluded"].items():
                st.markdown(f"**{tool}**")
                for reason in reasons:
                    st.markdown(f"- {reason}")

    with st.expander("Benchmark evidence"):
        segments = result["segments"]
        st.write(
            "On each benchmark dataset, the eight tools were ranked from 1 (best) to 8. The "
            "table shows their average rank on the datasets used for your order. A high "
            "segment is the group of benchmark datasets that share a characteristic of your "
            "dataset, such as more than 10,000 rows. If your dataset belongs to no high "
            "segment, all 30 datasets are used. If it belongs to several, the mean of their "
            "average ranks decides the order.")
        st.write(
            "The five recommended tools form the top group: in the Friedman and Nemenyi test "
            "over all 30 datasets, none of them ranked significantly lower than the best tool, "
            "AutoGluon. The other tools did.")
        shown = result["ranked"] + result["alternatives"]
        table = {"Tool": list(shown),
                 "Group": ["Recommended"] * len(result["ranked"])
                          + ["Other"] * len(result["alternatives"])}
        for segment in segments:
            table[SEGMENT_LABELS[segment]] = [f"{RANKS[segment][t]:.2f}" for t in shown]
        if len(segments) > 1:
            table["Mean"] = [f"{result['scores'][t]:.2f}" for t in shown]
        st.dataframe(table, hide_index=True)
        st.caption(
            "The benchmark used a 15-minute budget, 8 CPU cores, and 32 GB of memory. "
            "Ranks are pooled over the task types. Each high segment contains only five to ten "
            "datasets, so the differences are descriptive. All notes and warnings refer to the "
            "tool versions and settings of the benchmark."
        )


def questionnaire():
    st.subheader("Check whether this guide fits your task", anchor=False)
    st.info(
        "You can use this guide if all of the following apply:\n"
        "- Your data are in one table, including the column you want to predict.\n"
        "- Your task uses tabular features, without free-text inputs such as comments or "
        "descriptions.\n"
        "- Your data are not a time series. In a time series, the rows follow each other in "
        "time, and future values are predicted from past ones, such as next month's sales from "
        "the sales of previous months.\n"
        "- Your table has up to 100,000 rows and 1,000 input columns, excluding the column you "
        "want to predict.\n\n"
        "These limits reflect the datasets used to test the framework. Recommendations are "
        "available for tasks within this scope."
    )
    if not st.checkbox("My task meets these conditions.", key="scope"):
        st.caption("Tick the box to continue. If your task does not meet these conditions, the "
                   "guide cannot recommend a tool, because the tools were not compared on such "
                   "tasks.")
        return
    # Confirming the conditions S1 to S4 (Table 4.10) answers the internal scope
    # checks of recommendations.py.
    answers.update({"S1": "Yes", "S2": "No", "S3": "No"})

    st.subheader("Environment", anchor=False)
    ask("E1", "Which operating system will you use?", ("Linux", "macOS", "Windows"),
        help_text="The benchmark ran on Linux, so Linux has no restrictions. On macOS and "
                  "Windows, some tools need a container or additional system software.")
    ask("E2", "Can containers be used?",
        help_text="A container runs a separate Linux system on your computer, for example "
                  "with Docker Desktop. If you do not know or need permission from IT to "
                  "install Docker, answer No.")
    if answers["E2"] == "No":
        ask("E3", "Can you install system software, such as Java, OpenMP, or build tools?",
            help_text="This means software installed on the computer itself, outside Python, "
                      "which often requires administrator rights. If you cannot install "
                      "programs yourself, answer No.")
    ask("E4", "How much memory (RAM) does the computer have?", MEMORY,
        help_text="The working memory (RAM) of the computer. On a Mac, see Apple menu, "
                  "About This Mac. On Windows, see Settings, System, About.")

    st.subheader("Dataset characteristics", anchor=False)
    st.caption("If you are not sure about an answer, choose No. Each group of questions has a "
               "short Python script that helps you check the answers.")
    ask("D1", "What should be predicted?", TASK_TYPES,
        "Two classes: binary classification. More than two classes: multiclass "
        "classification. A number: regression.")
    classification = answers["D1"] in TASK_TYPES[:2]

    st.markdown("**Size and content of the data**")
    st.caption("These answers decide the order of the recommended tools.")
    with st.expander("How to check these answers with Python"):
        st.code(CHECK_SIZE, language="python")
        st.caption("Categories stored as numeric codes are counted as numeric here.")
    ask("D2", "Does the table have more than 10,000 rows?",
        help_text="Each row is one example, such as one patient or one customer.")
    ask("D3", "Does the table have more than 100 predictor columns?",
        help_text="Predictor columns are all columns used to make the prediction, that is, "
                  "every column except the one to predict.")
    ask("D4", "Are more than one third of the predictors categorical?",
        help_text="Categorical columns contain a limited set of categories instead of "
                  "measurements, such as region, product type, or payment method.")
    ask("D5", "Do more than one third of the rows have missing values?",
        help_text="A row counts if any of its cells is empty.")
    if classification:
        ask("D6", "Is the largest class more than ten times larger than the smallest?",
            help_text="For example, 1,000 healthy patients and fewer than 100 sick ones.")

    st.markdown("**Format of the data**")
    st.caption("These answers do not change the order. They only add preparation tips and "
               "warnings. The tool cards describe all known problems in any case.")
    with st.expander("How to check these answers with Python"):
        st.code(CHECK_FORMAT, language="python")
        st.caption("Categories stored as numeric codes and identifier columns cannot be detected "
                   "reliably by code, so check them by looking at the columns.")
    if classification:
        ask("D7", "Are the class labels stored as numbers?",
            help_text="For example, 0 and 1 used as labels for two categories.")
    ask("D8", "Are categories stored as numeric codes?",
        help_text="For example, regions represented by codes 1, 2, and 3.")
    ask("D9", "Are there identifier columns, such as names or ID numbers?")
    ask("D10", "Are there empty or constant columns?",
        help_text="An empty column has no values. A constant column has the same value in "
                  "every row.")
    if classification:
        ask("D11", "Does any class have fewer than ten examples?",
            help_text="For example, a diagnosis that occurs in fewer than ten rows.")

    st.subheader("Usage context", anchor=False)
    ask("U1", "Do you prefer tools that accept the table as it is?",
        help_text="That is, without declaring column types or recoding class labels. This is "
                  "a preference, not a requirement. Tools that need such a step are "
                  "still recommended, but placed after the tools that need none, and the step "
                  "is flagged.")
    ask("U2", "How much time is available for training?", TIME)

    # Unanswered questions in the order shown. A question that appears after an
    # earlier answer changes (for example E3 when containers cannot be used) is
    # named here, so that users can find it.
    missing = [LABELS[key] for key, value in answers.items() if value is None]
    completed = not missing
    button_column, status_column = st.columns([1, 2], vertical_alignment="center")
    with button_column:
        clicked = st.button("Get recommendations", type="primary", disabled=not completed)
    if clicked:
        st.session_state["submitted_answers"] = dict(answers)
    submitted = st.session_state.get("submitted_answers")
    with status_column:
        if len(missing) == 1:
            st.info(f"One question left: {missing[0]}")
        elif missing:
            st.info(f"{len(missing)} questions left. First unanswered: {missing[0]}")
        elif submitted == answers:
            st.caption("Your recommendations are shown below.")
        elif submitted is not None:
            st.caption("Select Get recommendations again to apply your updated answers.")
        else:
            st.caption("All questions are answered. Select Get recommendations.")
    if completed and submitted == answers:
        task = "classification" if classification else "regression"
        show_results(recommend(answers), task, answers)

with guide:
    questionnaire()
