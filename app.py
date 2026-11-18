"""Streamlit interface of the AutoML Selection Guide (thesis Section 4.4).

Run with:  streamlit run app.py
The questionnaire follows Tables 4.10 to 4.13, the rules are in
recommendations.py, and the tool cards follow Appendix D.
"""

import json
from pathlib import Path

import streamlit as st

from recommendations import (LOW_PREP, MEMORY, RANKS, SEGMENT_LABELS, TASK_TYPES,
                             TIME, TOP, recommend, scope_issues)

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


def show_card(tool):
    for label, value in CARDS[tool].items():
        st.markdown(f"**{label}**  \n{value}")
    st.markdown(f"[Open official documentation]({DOCS[tool]})")


with library:
    st.caption(
        "These cards describe the versions and settings tested in the thesis. Resource use was "
        "observed with a budget of 15 minutes, 8 CPU cores, and 32 GB of memory. It is not a "
        "minimum requirement or a guaranteed upper limit."
    )
    for tool in CARDS:
        with st.expander(tool):
            show_card(tool)


answers = {}


def ask(key, label, options=YES_NO, help_text=None):
    answers[key] = st.radio(label, options, index=None, key=key, help=help_text,
                            horizontal=len(options) <= 3)


def position_reason(tool, result):
    segments = result["segments"]
    score = result["scores"][tool]
    if segments == ["all"]:
        basis = "over all 30 benchmark datasets"
    elif len(segments) == 1:
        basis = f"in the high segment ({SEGMENT_LABELS[segments[0]].lower()})"
    else:
        basis = None
    if basis is None:
        text = (f"Mean rank {score:.2f} over the {len(segments)} high segments that apply "
                "to your dataset.")
    else:
        text = f"Average rank {score:.2f} {basis}."
    if result["prefer_low_prep"]:
        if tool in LOW_PREP:
            text += (" Placed ahead of the tools that need preparation, because it accepted "
                     "the table without type declarations or label recoding in the tested setup.")
        else:
            text += (" Placed after AutoGluon and FLAML, which accepted the table without type "
                     "declarations or label recoding.")
    return text


def show_results(result):
    st.header("Your recommendations", anchor=False)
    st.caption(
        "The top-group tools did not differ significantly in the benchmark, so the order is "
        "indicative, not a prediction of the best tool for your data. Treat it as a shortlist "
        "to evaluate on your own data. The benchmark used a 15-minute budget, 8 CPU cores, "
        "and 32 GB of memory."
    )
    for message in result["general"]:
        st.warning(message)

    st.subheader("Top group", anchor=False)
    if not result["ranked"]:
        st.info("Your environment excludes every tool in the top group. "
                "Review the other available tools below.")
    for position, tool in enumerate(result["ranked"], 1):
        with st.container(border=True):
            st.markdown(f"**{position}. {tool}**")
            st.write(position_reason(tool, result))
            for note in result["notes"].get(tool, []):
                st.markdown(f"- {note}")
            with st.expander(f"View {tool} tool card"):
                show_card(tool)

    if result["alternatives"]:
        st.subheader("Other available tools", anchor=False)
        for tool in result["alternatives"]:
            highlighted = tool in result["short_budget"]
            label = f"{tool} (possible alternative for less than 15 minutes)" if highlighted else tool
            with st.expander(label, expanded=highlighted):
                st.write(f"Average rank {RANKS['all'][tool]:.2f} over all 30 datasets. Ranked "
                         "significantly lower than the best tool in the overall comparison.")
                if highlighted:
                    st.write(f"Finished well before the 15-minute budget in the benchmark "
                             f"({FIT_TIMES[tool]}). These times do not guarantee a similar "
                             "time on a new dataset.")
                for note in result["notes"].get(tool, []):
                    st.markdown(f"- {note}")
                show_card(tool)

    if result["excluded"]:
        with st.expander("Tools excluded by your environment", expanded=True):
            for tool, reasons in result["excluded"].items():
                st.markdown(f"**{tool}**")
                for reason in reasons:
                    st.markdown(f"- {reason}")

    with st.expander("Benchmark evidence"):
        segments = result["segments"]
        st.write("Average ranks of the recommended top-group tools on the benchmark datasets "
                 "used for the order (1 = best). If several segments apply, their mean "
                 "decides the order.")
        shown = result["ranked"]
        table = {"Tool": list(shown)}
        for segment in segments:
            table[SEGMENT_LABELS[segment]] = [f"{RANKS[segment][t]:.2f}" for t in shown]
        if len(segments) > 1:
            table["Mean"] = [f"{result['scores'][t]:.2f}" for t in shown]
        st.dataframe(table, hide_index=True)
        st.caption(
            "Ranks are pooled over the task types. Each high segment contains only five to ten "
            "datasets, so the differences are descriptive. All notes and warnings refer to the "
            "tool versions and settings of the benchmark."
        )


def questionnaire():
    st.subheader("Task scope", anchor=False)
    ask("S1", "Is the data a single table that contains the column to predict?")
    ask("S2", "Does the table contain free text, such as comments or descriptions, or data "
              "ordered in time, such as daily sales?")
    ask("S3", "Does the table have more than 100,000 rows or more than 1,000 columns besides "
              "the column to predict?")
    issues = scope_issues(answers)
    if issues:
        st.warning("Your task lies outside the scope of the framework, so no tool is recommended.")
        for issue in issues:
            st.markdown(f"- {issue}")
        return
    if None in answers.values():
        st.info("Answer the scope questions to continue.")
        return

    st.subheader("Environment", anchor=False)
    ask("E1", "Which operating system will the tool run on?", ("Linux", "macOS", "Windows"),
        help_text="The benchmark ran on Linux, so Linux has no restrictions. On macOS and "
                  "Windows, some tools need a container or additional system software.")
    ask("E2", "Can containers be used?",
        help_text="For example Docker or Apple container, to run a Linux environment.")
    if answers["E2"] == "No":
        ask("E3", "Can additional system software be installed, such as Java, an OpenMP library, "
                  "or build tools?")
    ask("E4", "How much memory is available?", MEMORY)

    st.subheader("Dataset characteristics", anchor=False)
    ask("D1", "What should be predicted?", TASK_TYPES,
        "Two classes: binary classification. More than two classes: multiclass "
        "classification. A number: regression.")
    ask("D2", "Does the table have more than 10,000 rows?")
    ask("D3", "Does the table have more than 100 predictor columns?")
    ask("D4", "Are more than one third of the predictors categorical?",
        help_text="Examples include region, product type, and payment method.")
    ask("D5", "Do more than one third of the rows have at least one missing value?")
    classification = answers["D1"] in TASK_TYPES[:2]
    if classification:
        ask("D6", "Is the most frequent class more than ten times as frequent as the least "
                  "frequent class?")
        ask("D7", "Are the class labels stored as numbers?",
            help_text="For example, 0 and 1 used as labels for two categories.")
    ask("D8", "Are categories stored as numeric codes?",
        help_text="For example, regions represented by codes 1, 2, and 3.")
    ask("D9", "Does the table contain identifier columns, such as names or ID numbers?")
    ask("D10", "Are there empty or constant columns?")
    if classification:
        ask("D11", "Does any class have fewer than ten examples?")

    st.subheader("Usage context", anchor=False)
    ask("U1", "Should the tool accept the table without declaring column types or recoding "
              "class labels?")
    ask("U2", "How much time is available for training?", TIME)

    completed = None not in answers.values()
    # One status line above the button, always present, so that the button and
    # everything below it stay in place when the last question is answered.
    status = st.empty()
    if st.button("Get recommendations", type="primary", disabled=not completed):
        st.session_state["submitted_answers"] = dict(answers)
    submitted = st.session_state.get("submitted_answers")
    if not completed:
        status.caption("Answer all visible questions to get recommendations.")
    elif submitted == answers:
        status.caption("Your recommendations are shown below.")
    elif submitted is not None:
        status.caption("Select Get recommendations again to apply your updated answers.")
    else:
        status.caption("All questions are answered. Select Get recommendations.")
    if completed and submitted == answers:
        show_results(recommend(answers))

with guide:
    questionnaire()
