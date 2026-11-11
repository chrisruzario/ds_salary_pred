# Data Science Salary Explorer and Estimator

**[Try the live salary estimator](https://chrisruzariosalrypred.streamlit.app/)**

A work-in-progress Python project exploring historical US job listings and preparing data for salary estimation. The target is the midpoint of a listed annual salary range in USD, not an employee's actual earnings.

## Current status — 11 November 2026

Completed data preparation, local model comparison, Ridge tuning, held-out evaluation and error analysis. Results below were recorded from the author's local Spyder run. **The committed `run.py` includes model saving and reload verification, and `predict.py` makes standalone predictions without retraining.** The author successfully generated the outputs locally and confirmed that reloaded predictions match. The fitted model and its training metadata are committed for deployment; test predictions remain excluded. Running `run.py` recreates the outputs. The Streamlit interface is deployed publicly on Streamlit Community Cloud and loads the saved pipeline without retraining. The default online prediction was verified on 11 November 2026: **$148,053**, matching the local result.

## Data source and attribution

This project uses the existing `glassdoor_jobs.csv` from [PlayingNumbers/ds_salary_proj](https://github.com/PlayingNumbers/ds_salary_proj), associated with Ken Jee's [salary project video series](https://www.youtube.com/playlist?list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t). The dataset was not personally scraped for this project. The tutorial is the learning reference; this implementation adds explicit salary review and description-group separation when splitting data.

Download the source CSV from [the original repository](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/glassdoor_jobs.csv). The dataset is not redistributed here; consult the source's applicable reuse terms. This repository's code license does not grant rights to third-party data.

## Completed work

- Removed the exported row-index column and identified missing salaries, including the `-1` marker.
- Excluded hourly listings rather than assuming annual working hours.
- Parsed annual minimum, maximum and midpoint salary values.
- Removed exact duplicate rows and inspected extreme salary estimates.
- Set aside one NPD Data Scientist listing with an unverified $15K–$16K estimate. The job description requests 6+ years of experience but does not verify the pay amount or period. This is a review decision, not proof of an error; the row remains in `salary_review` in memory.
- Removed trailing ratings from company names.
- Created keyword indicators for Python, SQL, Excel, Power BI, AWS and Spark.
- Extracted state codes and created initial job-family and seniority classifications.
- Compared salary midpoints by job family and seniority.
- Grouped identical descriptions after normalising case and whitespace, keeping each group on one side of the train/test split.

## Data preparation results

These counts were observed in the local run using the tutorial CSV.

| Step | Rows remaining |
|---|---:|
| Original data | 956 |
| Remove 214 listings without salary | 742 |
| Exclude 24 hourly listings | 718 |
| Remove 261 exact duplicate rows | 457 |
| Set aside one unverified salary listing | 456 |

All 718 non-hourly salary ranges parsed successfully. No non-positive minimum salaries or reversed salary ranges were found. Eight retained rows share four repeated-description groups.

`GroupShuffleSplit(test_size=0.20, random_state=42)` produced **365 training rows**, **91 test rows**, and **zero shared description groups**.

## Initial findings

| Job family | Listings | Median annual salary midpoint (USD) |
|---|---:|---:|
| Data Science | 204 | 112,250 |
| Machine Learning | 17 | 107,500 |
| Data Engineering | 75 | 98,500 |
| Data Architecture / Modeling | 5 | 91,500 |
| Other Science | 68 | 91,000 |
| Other | 7 | 87,500 |
| Analytics Consulting | 5 | 66,500 |
| Analysis / Analytics | 75 | 63,000 |

Python appears in 259 descriptions and SQL in 243. These are keyword counts in this historical sample, not estimates of present-day market demand or causal skill premiums.

## Run locally

Use Python 3.12 in a dedicated environment. Install dependencies into that same environment:

```bash
python -m pip install -r requirements.txt
```

Place `glassdoor_jobs.csv` alongside `run.py`, then run:

```bash
python run.py
```

In Spyder, select the corresponding Python environment and run `run.py`. The script resolves the CSV relative to its own location, prints the analysis and creates `df`, `salary_review`, `train_df` and `test_df` in memory. It also compares models, tunes Ridge, evaluates the selected pipeline and creates `best_model` and `test_results` in memory. It saves three files under `outputs/`:

| File | Contents |
|---|---|
| `salary_ridge_pipeline.joblib` | Fitted preprocessing and Ridge pipeline, trained on the 365 training rows |
| `test_predictions.csv` | Held-out predictions, actual midpoints and errors |
| `model_metadata.json` | Model settings, features, metrics and Python/package versions |

The script reloads the saved pipeline and checks that its predictions match using `numpy.testing.assert_allclose`. The reload check passed in the author's local run. Re-running training overwrites these output files.

After running `run.py` once, make a standalone prediction:

```bash
python predict.py
```

This loads the local pipeline without retraining. The included hypothetical Senior Data Science role in California (Python, SQL and AWS mentioned) produced **$148,053** in the author's run. This is a historical salary-midpoint estimate, not current salary guidance. Only load trusted joblib files and reuse the recorded package versions when loading an existing model. Direct dependencies in `requirements.txt` are pinned to the recorded training versions plus Streamlit 1.67.0; transitive dependencies are not fully locked.

## Browser app

The repository includes the fitted model, so retraining and the source CSV are not required to launch the interface. From the project folder, run:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal (usually http://localhost:8501) and keep that terminal running. Select a job family, seniority, one of the 12 included state/district codes, and skills mentioned in the listing, then click **Estimate salary**. The app shows the predicted annual salary midpoint, held-out MAE and submitted inputs.

The author checked the following locally with Streamlit 1.67.0:

- Default Senior Data Science role in CA, with Python, SQL and AWS: **$148,053**, matching `predict.py`.
- Changing only seniority to Unspecified: **$125,071**.
- Restoring Senior: **$148,053** again.

These are functional checks, not new model-validation results. Unspecified means seniority was not identified in the title; it does not mean junior. The displayed test MAE is the recorded evaluation result, not a confidence interval for an individual prediction.

The app loads the committed `outputs/salary_ridge_pipeline.joblib`; its training metadata is in `outputs/model_metadata.json`. The public demo is available at [chrisruzariosalrypred.streamlit.app](https://chrisruzariosalrypred.streamlit.app/).

### Deploy on Streamlit Community Cloud

Sign in at https://share.streamlit.io/ and create an app with:

- Repository: `chrisruzario/ds_salary_pred`
- Branch: `main`
- Main file: `app.py`
- Advanced settings → Python: `3.12`

The deployed app loaded successfully and returned **$148,053** for the default inputs in a browser check on 11 November 2026, matching the local result. This verifies the hosted example, not every possible input combination. Package versions were taken from the supplied model metadata.

## Limitations

- Historical, non-representative US listings; not current salary guidance or an Irish market dataset.
- Salary ranges include platform estimates and employer estimates, not verified earnings.
- Small job-family/seniority groups have unstable summary statistics.
- Keyword features and title classifications are heuristic; for example, the verb "excel" may produce a false skill match.
- Grouping prevents exact normalised descriptions crossing the split, but does not prevent all company overlap or near-duplicate text.
- Exploratory summaries were viewed before the split. The test set was held aside during model fitting and tuning, but is not an entirely unseen external validation dataset.
- The salary-review exclusion should be revisited with a sensitivity check.

## Modelling results — local run

Inputs: job family, seniority, state and six skill indicators. Salary-derived fields were excluded from predictors. Categorical values were one-hot encoded with unknown categories ignored; skill indicators were standardised inside the preprocessing pipeline.

Five-fold GroupKFold cross-validation used only the 365 training rows, keeping repeated-description groups together. Preprocessing was fitted within each fold.

| Model | Mean cross-validation MAE (USD) |
|---|---:|
| Median baseline | 27,111 |
| Ridge | 18,845 |
| Random Forest | 19,261 |

Random Forest used 300 trees, a minimum leaf size of 3 and random seed 42. Ridge used the lsqr solver. GridSearchCV compared alpha values 0.1, 1, 10, 100 and 1000; **alpha=1** remained best. The best pipeline was refitted on all training rows before test evaluation.

### Held-out test results

| Metric | Result |
|---|---:|
| Test rows | 91 |
| Baseline MAE | $29,692 |
| Ridge MAE | $20,943 |
| Ridge RMSE | $28,599 |
| Ridge R² | 0.427 |
| MAE improvement over baseline | 29.5% |
| Median absolute error | $13,722 |
| 90th percentile absolute error | $44,396 |
| Predictions within $10,000 | 35.2% |
| Predictions within $20,000 | 59.3% |
| Mean signed error (prediction minus target) | −$7,067 |
| Shared description groups across train/test | 0 |

The model underpredicted on average. Several expensive or specialised roles had large errors: the Gallup Senior Data Scientist listing had a $237,500 midpoint and an approximately $127,053 prediction. Conversely, the MassMutual Data Science Project Manager listing had a $68,500 midpoint and an approximately $136,303 prediction.

These observations suggest limitations in broad title/skill features, but do not establish the causes of each error. Company identity, industry and detailed experience requirements were not model inputs. MAE is an average error, not a guaranteed prediction interval. This is a learning prototype using historical estimated salary ranges, not a validated salary-negotiation tool. No model changes were selected using the reported test results.

## Next steps

1. Verify the workflow in a fresh environment using the recorded versions.
2. Assess the salary-review exclusion in a separate sensitivity study without treating the already-inspected test set as new validation data.
3. Expand interface checks across additional job families, seniority levels and locations.
