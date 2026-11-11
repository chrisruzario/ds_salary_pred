# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:33:00 2026

@author: chrisruzario
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Data Salary Estimator",
    layout="centered",
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "salary_ridge_pipeline.joblib"
)

st.title("Data Salary Estimator")
st.write("Explore estimated annual salary midpoints for US data roles.")
st.caption(
    "Portfolio demo using historical job listings. "
    "These estimates are not current salary guidance."
)

if not MODEL_PATH.exists():
    st.error(
        "Saved model not found. Run run.py to generate "
        "outputs/salary_ridge_pipeline.joblib, then refresh this page."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("salary_form"):
    job_family = st.selectbox(
        "Job family",
        [
            "Data Science",
            "Data Engineering",
            "Analysis / Analytics",
            "Machine Learning",
            "Data Architecture / Modeling",
            "Analytics Consulting",
            "Other Science",
            "Other",
        ],
    )

    seniority = st.selectbox(
        "Seniority",
        [
            "Unspecified",
            "Associate",
            "Junior / Entry / Intern",
            "Senior",
            "Lead / Principal / Staff",
            "Management",
        ],
        index=3,
    )

    state = st.selectbox(
        "State / district",
        [
            "CA", "MA", "NY", "VA", "MD", "IL",
            "PA", "TX", "WA", "NJ", "DC", "CO",
        ],
        help="This initial demo includes a selection of dataset locations.",
    )

    st.subheader("Skills mentioned in the job listing")
    st.caption("Select skills required by the role, rather than your own skills.")

    left, right = st.columns(2)

    with left:
        python = st.checkbox("Python", value=True)
        sql = st.checkbox("SQL", value=True)
        excel = st.checkbox("Excel")

    with right:
        power_bi = st.checkbox("Power BI")
        aws = st.checkbox("AWS", value=True)
        spark = st.checkbox("Spark")

    submitted = st.form_submit_button("Estimate salary")

if submitted:
    job = pd.DataFrame([{
        "job_family": job_family,
        "seniority": seniority,
        "state": state,
        "python": int(python),
        "sql": int(sql),
        "excel": int(excel),
        "power_bi": int(power_bi),
        "aws": int(aws),
        "spark": int(spark),
    }])

    prediction = float(model.predict(job)[0])

    st.metric(
        label="Predicted annual salary midpoint (USD)",
        value=f"${prediction:,.0f}",
    )

    st.caption(
        "Held-out test MAE: $20,943 across 91 listings. "
        "This is average absolute error, not a confidence interval "
        "for this prediction."
    )

    with st.expander("View prediction inputs"):
        st.dataframe(job, hide_index=True)
