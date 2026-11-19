# AutoML Selection Guide

Streamlit app of the AutoML tool selection framework described in Section 4.4 of the master thesis *A Framework for Using AutoML* (Tatiana Korzineva, FHNW, 2026).

## Run locally on macOS

Open Terminal in this folder and run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address localhost
```

Open the local URL shown in Terminal. Press Control-C in Terminal to stop the app.

## Check the selection logic

```sh
python -m pip install pytest
python -m pytest test_recommendations.py
```

The tests check every rule of Tables 4.10 to 4.13 and the worked example in Section 4.4.2 (binary task with more than 100 predictors).

## Files

| File | Content | Source in the thesis |
|---|---|---|
| `app.py` | Questionnaire and result page (interface only) | Section 4.4.1, 4.4.3 |
| `recommendations.py` | Selection logic in five steps | Section 4.4.2, Tables 4.10 to 4.13 |
| `data/mean_ranks.csv` | Average ranks overall, per task type and per high segment | Table 4.8 |
| `data/tool_cards.json` | Tool cards | Appendix D |
| `test_recommendations.py` | Checks of the selection logic | |

`mean_ranks.csv` is derived from `results/benchmark/tables/thesis/results_by_dataset.csv` in the code package; the values were checked against that table. If the thesis tables change, update these files and run the tests again.

## Behaviour

- The top group (AutoGluon, auto-sklearn, MLJAR AutoML, FLAML, H2O AutoML) comes from the overall Friedman and Nemenyi comparison over all 30 datasets.
- The top group is ordered by the average rank in each high segment that applies (D2 to D6), averaged over these segments; if the dataset is low on every characteristic, the ranks over all 30 datasets are used. Exact ties keep the overall order.
- U1 is a preference. If it is Yes, the tools that need no column-type declaration or label recoding are placed first (AutoGluon and FLAML, and MLJAR AutoML for multiclass and regression), and the ranks decide the order within the two parts. The steps of the other tools are flagged.
- Not sure (D2 to D11): for D2 to D6 it counts as No for the order, and the output names these characteristics. For D7 to D11 the notes are shown as for Yes.
- Questions D7 to D11 only add notes and warnings; they never exclude a tool or change the order.
- Containers are asked first (E2). The question on system software (E3) is only asked if containers cannot be used.
- Internet access is not asked. It concerns only Lightwood, which gets a note that it needs internet during training unless its NLTK language data are installed beforehand.
- No data is uploaded; the app uses only the answers.
