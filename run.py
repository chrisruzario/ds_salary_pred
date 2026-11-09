# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 12:47:54 2026

@author: chrisruzario
"""

from pathlib import Path

import pandas as pd

df = pd.read_csv(
    Path(__file__).resolve().parent / "glassdoor_jobs.csv"
)

print("Rows and columns:", df.shape)
print(df.head())
print(df.columns.tolist())

# Remove the saved row-index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Inspect salary values before cleaning them
print("\nSalary examples:")
print(df["Salary Estimate"].head(10))

# Count missing salaries, including the scraper's -1 marker
salary_text = df["Salary Estimate"].astype("string").str.strip()
missing_salary = (
    salary_text.isna()
    | salary_text.eq("-1")
    | salary_text.eq("")
).fillna(True)

print("\nRows without salary:", missing_salary.sum())
print("Rows with salary:", (~missing_salary).sum())

# Keep only rows with salary information
df = df.loc[~missing_salary].copy()

print("Rows remaining:", len(df))

# Inspect the different salary formats
print("\nSalary examples:")
print(df["Salary Estimate"].head(10).to_string(index=False))

# Check for hourly pay
hourly = df["Salary Estimate"].str.contains(
    "per hour", case=False, na=False
)

print("\nHourly salary rows:", hourly.sum())
print(df.loc[hourly, "Salary Estimate"].head().to_string(index=False))

# Exclude hourly listings
df = df.loc[~hourly].copy()

# Extract annual salary ranges such as $53K-$91K
salary_parts = df["Salary Estimate"].str.extract(
    r"\$(\d+(?:\.\d+)?)K\s*-\s*\$(\d+(?:\.\d+)?)K"
)

# Convert thousands into actual dollar amounts
df["salary_min"] = pd.to_numeric(salary_parts[0]) * 1000
df["salary_max"] = pd.to_numeric(salary_parts[1]) * 1000

# Calculate the midpoint of each salary range
df["salary_midpoint"] = (
    df["salary_min"] + df["salary_max"]
) / 2

# Check for formats our pattern didn't recognise
unparsed = df["salary_midpoint"].isna()

print("\nRows remaining:", len(df))
print("Unparsed salary rows:", unparsed.sum())

print("\nUnparsed examples:")
print(df.loc[unparsed, "Salary Estimate"].head(10).to_string(index=False))

print("\nConverted salaries:")
print(
    df[["Salary Estimate", "salary_min",
        "salary_max", "salary_midpoint"]].head()
)

# Count and remove exact duplicates
duplicate_count = df.duplicated().sum()
print("\nExact duplicate rows:", duplicate_count)

df = df.drop_duplicates().reset_index(drop=True)
print("Rows after removing duplicates:", len(df))

# Check for invalid salary ranges
invalid_salary = (
    (df["salary_min"] <= 0)
    | (df["salary_max"] < df["salary_min"])
)

print("Invalid salary ranges:", invalid_salary.sum())

# Summarise annual salary midpoints in dollars
print("\nSalary midpoint summary:")
print(df["salary_midpoint"].describe().round(2))

# Inspect the five lowest and highest salary midpoints
review_columns = [
    "Job Title",
    "Salary Estimate",
    "Company Name",
    "Location",
    "salary_midpoint",
]

print("\nFive lowest salaries:")
print(
    df.nsmallest(5, "salary_midpoint")[review_columns]
    .to_string(index=False)
)

print("\nFive highest salaries:")
print(
    df.nlargest(5, "salary_midpoint")[review_columns]
    .to_string(index=False)
)

# Inspect the unusually low salary listing
suspect = df["salary_midpoint"].eq(15500)

for _, row in df.loc[suspect].iterrows():
    print("\nJob title:", row["Job Title"])
    print("Salary:", row["Salary Estimate"])
    print("\nFull job description:")
    print(row["Job Description"])
    
# Flag this specific listing, rather than all low salaries
suspect = (
    df["Company Name"].str.split("\n").str[0].str.strip().eq("NPD")
    & df["Job Title"].str.strip().eq("Data Scientist")
    & df["salary_min"].eq(15000)
    & df["salary_max"].eq(16000)
)

salary_review = df.loc[suspect].copy()
salary_review["review_reason"] = (
    "Salary estimate cannot be verified from the description; "
    "role requires 6+ years of experience."
)

df = df.loc[~suspect].copy()

# Remove trailing ratings from company names
# Example: "NPD\n3.9" becomes "NPD"
df["company_name_clean"] = (
    df["Company Name"]
    .str.replace(r"\n\s*\d+(?:\.\d+)?\s*$", "", regex=True)
    .str.strip()
)

print("\nListings set aside:", len(salary_review))
print("Rows remaining:", len(df))

print("\nCleaned company names:")
print(
    df[["Company Name", "company_name_clean"]]
    .head()
    .to_string(index=False)
)
# Use empty text for missing descriptions
descriptions = df["Job Description"].fillna("")

# Patterns match whole words to reduce false matches
skill_patterns = {
    "python": r"\bpython\b",
    "sql": r"\bsql\b",
    "excel": r"\bexcel\b",
    "power_bi": r"\bpower\s*bi\b",
    "aws": r"\baws\b|\bamazon web services\b",
    "spark": r"\bspark\b",
}

for skill, pattern in skill_patterns.items():
    df[skill] = descriptions.str.contains(
        pattern,
        case=False,
        regex=True,
        na=False,
    ).astype(int)

print("\nNumber of listings mentioning each skill:")
print(df[list(skill_patterns)].sum().sort_values(ascending=False))

print("\nExamples:")
print(df[["Job Title", "python", "sql", "excel"]].head())

# Extract a two-letter state code at the end of the location
df["state"] = (
    df["Location"]
    .astype("string")
    .str.strip()
    .str.extract(r",\s*([A-Z]{2})$", expand=False)
)

print("\nTop 10 states by listing count:")
print(df["state"].value_counts().head(10))

# Inspect locations that don't match the expected format
print("\nLocations without a state code:")
print(
    df.loc[df["state"].isna(), "Location"]
    .value_counts(dropna=False)
)

print("\nNumber of unique job titles:")
print(df["Job Title"].nunique())

print("\nTop 25 job titles:")
print(df["Job Title"].value_counts().head(25).to_string())

# Inspect titles that may need separate categories
special_titles = df["Job Title"].str.contains(
    r"analyst|engineer|manager|director|scientist",
    case=False,
    na=False,
)

print("\nOther job titles:")
print(
    df.loc[~special_titles, "Job Title"]
    .value_counts()
    .to_string()
)

import re

def classify_job(title):
    title = str(title).lower()

    # Check specific roles before broader categories
    if re.search(r"\bmachine learning\b|\bml engineer\b", title):
        return "Machine Learning"

    if re.search(r"\bdata architect\b|\bdata modeler\b", title):
        return "Data Architecture / Modeling"

    if "architect" in title and "data" in title:
        return "Data Architecture / Modeling"

    if re.search(r"\bdata scientist\b|\bdata science\b", title):
        return "Data Science"

    if re.search(r"\bdata engineer\b|\bdata engineering\b", title):
        return "Data Engineering"

    if "consult" in title and re.search(r"\bdata\b|\banalytics\b", title):
        return "Analytics Consulting"

    if "analyst" in title or "analytics" in title:
        return "Analysis / Analytics"

    if "scientist" in title:
        return "Other Science"

    return "Other"


df["job_family"] = df["Job Title"].apply(classify_job)

print("\nJob family counts:")
print(df["job_family"].value_counts())

print("\nTitles classified as Other:")
print(
    df.loc[df["job_family"].eq("Other"), "Job Title"]
    .value_counts()
    .to_string()
)

def classify_seniority(title):
    title = str(title).lower()

    # Check leadership titles first
    if re.search(
        r"\b(?:director|vp|vice president|head|chief|manager)\b",
        title,
    ):
        return "Management"

    if re.search(r"\b(?:lead|principal|staff)\b", title):
        return "Lead / Principal / Staff"

    if re.search(r"\b(?:senior|sr)\b", title):
        return "Senior"

    if re.search(r"\b(?:junior|jr|entry[- ]level|intern|internship)\b", title):
        return "Junior / Entry / Intern"

    if re.search(r"\bassociate\b", title):
        return "Associate"

    return "Unspecified"


df["seniority"] = df["Job Title"].apply(classify_seniority)

print("\nSeniority counts:")
print(df["seniority"].value_counts())

print("\nExample titles from each group:")
for group, rows in df.groupby("seniority"):
    print(f"\n{group}:")
    print(rows["Job Title"].head(3).to_string(index=False))
    
# Summarise salary by job family
salary_by_family = (
    df.groupby("job_family")["salary_midpoint"]
    .agg(
        listings="count",
        median_salary="median",
        mean_salary="mean",
    )
    .round(0)
    .sort_values("median_salary", ascending=False)
)

print("\nAnnual salary midpoints by job family (USD):")
print(salary_by_family.to_string())

# Compare salaries by job family AND seniority
salary_by_level = (
    df.groupby(["job_family", "seniority"])["salary_midpoint"]
    .agg(
        listings="count",
        median_salary="median",
    )
    .round(0)
)

# Focus on the three largest data-related groups
main_families = [
    "Data Science",
    "Data Engineering",
    "Analysis / Analytics",
]

print("\nSalary by job family and seniority (USD):")
print(salary_by_level.loc[main_families].to_string())

# Normalise spacing and case to identify repeated descriptions
description_key = (
    df["Job Description"]
    .fillna("")
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Flag repeated, non-empty descriptions
repeated = (
    description_key.ne("")
    & description_key.duplicated(keep=False)
)

print("\nRows sharing a repeated description:", repeated.sum())
print(
    "Distinct repeated descriptions:",
    description_key.loc[repeated].nunique(),
)

print("\nExample listings with repeated descriptions:")
print(
    df.loc[
        repeated,
        ["Job Title", "company_name_clean", "Location", "salary_midpoint"],
    ].head(12).to_string(index=False)
)

from sklearn.model_selection import GroupShuffleSplit

# Group identical descriptions together.
# Give missing/empty descriptions their own group.
groups = description_key.copy()

empty_description = groups.eq("")
groups.loc[empty_description] = [
    f"missing_description_{i}"
    for i in df.index[empty_description]
]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_positions, test_positions = next(
    splitter.split(df, groups=groups)
)

train_df = df.iloc[train_positions].copy()
test_df = df.iloc[test_positions].copy()

# Verify that no description group appears in both sets
train_groups = set(groups.iloc[train_positions])
test_groups = set(groups.iloc[test_positions])

print("Training rows:", len(train_df))
print("Test rows:", len(test_df))

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

feature_columns = [
    "job_family", "seniority", "state",
    "python", "sql", "excel", "power_bi", "aws", "spark",
]

X_train = train_df[feature_columns]
y_train = train_df["salary_midpoint"]

training_groups = groups.iloc[train_positions]

baseline = DummyRegressor(strategy="median")
cv = GroupKFold(n_splits=5)

scores = cross_val_score(
    baseline,
    X_train,
    y_train,
    groups=training_groups,
    cv=cv,
    scoring="neg_mean_absolute_error",
)

# Convert negative scores into positive dollar errors
fold_mae = -scores

print("\nBaseline MAE for each fold:")
print(fold_mae.round(2))

print(f"\nAverage baseline MAE: ${fold_mae.mean():,.0f}")
print(f"Training median salary: ${y_train.median():,.0f}")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

categorical_features = ["job_family", "seniority", "state"]
numeric_features = [
    "python", "sql", "excel", "power_bi", "aws", "spark"
]

# Convert categories into numeric columns.
# Fit these transformations separately within each training fold.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "skills",
            StandardScaler(),
            numeric_features,
        ),
    ]
)

ridge_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", Ridge(alpha=1.0, solver="lsqr")),
])

ridge_scores = cross_val_score(
    ridge_pipeline,
    X_train,
    y_train,
    groups=training_groups,
    cv=cv,
    scoring="neg_mean_absolute_error",
)

ridge_mae = -ridge_scores
baseline_mae = fold_mae.mean()

print("\nRidge MAE for each fold:")
print(ridge_mae.round(2))

print(f"\nAverage Ridge MAE: ${ridge_mae.mean():,.0f}")
print(f"Baseline MAE: ${baseline_mae:,.0f}")

improvement = (
    (baseline_mae - ridge_mae.mean()) / baseline_mae
) * 100

print(f"Improvement over baseline: {improvement:.1f}%")

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

forest_pipeline = Pipeline([
    ("preprocessing", clone(preprocessor)),
    ("model", RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )),
])

forest_scores = cross_val_score(
    forest_pipeline,
    X_train,
    y_train,
    groups=training_groups,
    cv=cv,
    scoring="neg_mean_absolute_error",
)

forest_mae = -forest_scores

print("\nRandom Forest MAE for each fold:")
print(forest_mae.round(2))

comparison = pd.DataFrame({
    "Model": ["Median baseline", "Ridge", "Random Forest"],
    "CV MAE (USD)": [
        fold_mae.mean(),
        ridge_mae.mean(),
        forest_mae.mean(),
    ],
}).sort_values("CV MAE (USD)")

print("\nModel comparison — lower MAE is better:")
print(comparison.to_string(index=False, float_format="%.0f"))

from sklearn.model_selection import GridSearchCV

ridge_search = GridSearchCV(
    estimator=clone(ridge_pipeline),
    param_grid={
        "model__alpha": [0.1, 1, 10, 100, 1000]
    },
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
)

ridge_search.fit(
    X_train,
    y_train,
    groups=training_groups,
)

tuning_results = pd.DataFrame({
    "Alpha": ridge_search.cv_results_["param_model__alpha"],
    "CV MAE (USD)": -ridge_search.cv_results_["mean_test_score"],
}).sort_values("CV MAE (USD)")

print("\nRidge tuning results:")
print(tuning_results.to_string(index=False, formatters={"Alpha": "{:g}".format, "CV MAE (USD)": "{:,.0f}".format}))

print("\nBest alpha:", ridge_search.best_params_["model__alpha"])
print(f"Best CV MAE: ${-ridge_search.best_score_:,.0f}")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_test = test_df[feature_columns]
y_test = test_df["salary_midpoint"]

# GridSearchCV already fitted this pipeline on all training rows
best_model = ridge_search.best_estimator_
test_predictions = best_model.predict(X_test)

# Baseline learns its median only from the training data
final_baseline = DummyRegressor(strategy="median")
final_baseline.fit(X_train, y_train)
baseline_predictions = final_baseline.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
baseline_test_mae = mean_absolute_error(y_test, baseline_predictions)
test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5
test_r2 = r2_score(y_test, test_predictions)

print("\nHeld-out test results:")
print("Test rows:", len(y_test))
print(f"Baseline MAE: ${baseline_test_mae:,.0f}")
print(f"Ridge MAE: ${test_mae:,.0f}")
print(f"Ridge RMSE: ${test_rmse:,.0f}")
print(f"Ridge R²: {test_r2:.3f}")

improvement = (
    (baseline_test_mae - test_mae) / baseline_test_mae
) * 100

print(f"Test improvement over baseline: {improvement:.1f}%")

# Compare actual salary midpoints with model predictions
test_results = test_df[
    ["Job Title", "company_name_clean", "state", "seniority"]
].copy()

test_results["actual_salary"] = y_test.to_numpy()
test_results["predicted_salary"] = test_predictions

# Positive error = overprediction; negative = underprediction
test_results["error"] = (
    test_results["predicted_salary"] - test_results["actual_salary"]
)
test_results["absolute_error"] = test_results["error"].abs()

print("\nTen largest test errors:")
print(
    test_results.nlargest(10, "absolute_error")
    .to_string(index=False, float_format="%.0f")
)


absolute_errors = test_results["absolute_error"]

print("\nError distribution across all test listings:")
print(f"Median absolute error: ${absolute_errors.median():,.0f}")
print(f"90th percentile absolute error: ${absolute_errors.quantile(0.90):,.0f}")

print(
    "Predictions within $10,000:",
    f"{absolute_errors.le(10000).mean():.1%}"
)

print(
    "Predictions within $20,000:",
    f"{absolute_errors.le(20000).mean():.1%}"
)

print(
    "Mean signed error:",
    f"${test_results['error'].mean():,.0f}"
)
print("Shared description groups:", len(train_groups & test_groups))
