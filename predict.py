# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:09:25 2026

@author: chrisruzario
"""

from pathlib import Path
import joblib
import pandas as pd

model_path = (
    Path(__file__).resolve().parent
    / "outputs"
    / "salary_ridge_pipeline.joblib"
)

model = joblib.load(model_path)

# Hypothetical example, not an actual job listing
example_job = pd.DataFrame([{
    "job_family": "Data Science",
    "seniority": "Senior",
    "state": "CA",
    "python": 1,
    "sql": 1,
    "excel": 0,
    "power_bi": 0,
    "aws": 1,
    "spark": 0,
}])

prediction = model.predict(example_job)[0]

print(f"Predicted annual salary midpoint: ${prediction:,.0f}")
print("Historical US data estimate—not current salary guidance.")