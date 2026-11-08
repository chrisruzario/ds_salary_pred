# Data Science Salary Explorer and Estimator

A work-in-progress Python project exploring historical US job listings and preparing data for salary estimation. The target is the midpoint of a listed annual salary range in USD, not an employee's actual earnings.

## Current status — 8 November 2026

Completed data cleaning, feature engineering, exploratory analysis and a grouped train/test split. **No prediction model has been trained or evaluated yet.** There is no deployed app or API at this stage.

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

In Spyder, select the corresponding Python environment and run `run.py`. The script resolves the CSV relative to its own location, prints the analysis and creates `df`, `salary_review`, `train_df` and `test_df` in memory. It does not currently export those tables or train a model.

## Limitations

- Historical, non-representative US listings; not current salary guidance or an Irish market dataset.
- Salary ranges include platform estimates and employer estimates, not verified earnings.
- Small job-family/seniority groups have unstable summary statistics.
- Keyword features and title classifications are heuristic; for example, the verb "excel" may produce a false skill match.
- Grouping prevents exact normalised descriptions crossing the split, but does not prevent all company overlap or near-duplicate text.
- Exploratory summaries were viewed before the split. The test set is held aside for modelling, but is not an entirely unseen external validation dataset.
- The salary-review exclusion should be revisited with a sensitivity check.

## Next steps

1. Establish a median-salary baseline using grouped cross-validation within the training set.
2. Build preprocessing pipelines and compare regression models.
3. Keep salary-derived fields out of predictor inputs to avoid target leakage.
4. Evaluate the selected model on the held-out test set and document errors honestly.
5. Add an interface or API after the modelling workflow is validated.
