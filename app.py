import os
import io
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from docx import Document 

from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv

# Optional imports
try:
    import openai  # openai==0.28.1 syntax
except Exception:
    openai = None

try:
    import boto3
except Exception:
    boto3 = None

from docx import Document
from docx.shared import Inches

# ----------- Config ----------- 
st.set_page_config(page_title="Sustainably – Impact MVP", page_icon="🌍", layout="wide")
load_dotenv()

# User can input OpenAI API key here
OPENAI_API_KEY = st.text_input("Enter your OpenAI API key", type="password")  # Let user input API key

OPENAI_MODEL = "gpt-4"  # You can set your available model here like gpt-4 or gpt-3.5-turbo

# ----------- Constants ----------- 
IMF_COLUMNS = [
    "Outputs Name",
    "Outputs Description",
    "Indicator Name",
    "Indicator Description",
    "Unit of Measure"
]

DEFAULT_OUTPUTS = [
    {"Outputs Name": "Impact Reports Generated", "Outputs Description": "AI-powered reports...", "Indicator Name": "Number of reports produced", "Indicator Description": "Tracks how many...", "Unit of Measure": "Count (#)"},
    {"Outputs Name": "Stakeholder Feedback Collected", "Outputs Description": "Structured input...", "Indicator Name": "% of stakeholders providing feedback", "Indicator Description": "Measures the proportion...", "Unit of Measure": "Percentage (%)"},
    # Include other outputs as previously defined
]

# ----------- Helpers ----------- 

def read_data(upload):
    if upload is None:
        return None, None
    name = upload.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(upload)
        return df, "CSV"
    if name.endswith(".xlsx") or name.endswith(".xls"):
        xls = pd.ExcelFile(upload)
        return xls, "XLSX"
    st.error("Unsupported file type. Please upload CSV or Excel.")
    return None, None

def load_sheet(xls, sheet_name=None):
    if isinstance(xls, pd.DataFrame):
        return xls
    if isinstance(xls, pd.ExcelFile):
        if sheet_name is None:
            sheet_name = xls.sheet_names[0]
        return xls.parse(sheet_name)
    return None

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

# Predefined prompts for OpenAI to generate executive summaries
def llm_summary(prompt: str) -> str:
    if OPENAI_API_KEY and openai is not None:
        try:
            openai.api_key = OPENAI_API_KEY
            # 0.28.1 ChatCompletion format
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":"You are an expert social impact analyst."},
                          {"role":"user","content": prompt}],
                temperature=0.2,
                max_tokens=300,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"(LLM unavailable; using fallback summary.)"
    return "(Narrative) Sustainably ingested the provided dataset, computed core KPIs, and aligned outputs to recognized frameworks."

def compute_kpis(df, mappings, params):
    metrics = {}
    if df is None:
        return metrics

    dff = df.copy()
    if mappings.get("satisfaction") and mappings["satisfaction"] in dff.columns:
        dff["__satisfaction__"] = pd.to_numeric(dff[mappings["satisfaction"]], errors="coerce")
    else:
        dff["__satisfaction__"] = np.nan

    if mappings.get("report_id") and mappings["report_id"] in dff.columns:
        metrics["reports_produced"] = int(dff[mappings["report_id"]].nunique())
    else:
        metrics["reports_produced"] = int(len(dff))  # fallback

    if mappings.get("participant_id") and mappings["participant_id"] in dff.columns:
        metrics["participants_trained"] = int(dff[mappings["participant_id"]].nunique())
    else:
        metrics["participants_trained"] = int(len(dff))

    if mappings.get("program_completed") and mappings["program_completed"] in dff.columns:
        metrics["templates_completed"] = int(dff[mappings["program_completed"]].sum())
    else:
        metrics["templates_completed"] = 0

    if dff["__satisfaction__"].notna().any():
        metrics["avg_satisfaction"] = round(float(dff["__satisfaction__"].mean()), 2)
    else:
        metrics["avg_satisfaction"] = None

    baseline_hours = params.get("baseline_hours", 6.0)
    automated_hours = params.get("automated_hours", 2.0)
    hourly_rate = params.get("hourly_rate", 35.0)

    avg_hours_saved = max(baseline_hours - automated_hours, 0.0)
    metrics["avg_hours_saved_per_report"] = round(float(avg_hours_saved), 2)
    total_hours_saved = avg_hours_saved * metrics["reports_produced"]
    metrics["total_hours_saved"] = round(float(total_hours_saved), 2)
    metrics["money_saved"] = round(float(total_hours_saved * hourly_rate), 2)

    return metrics

def make_kpi_bar(metrics: dict) -> BytesIO:
    keys = ["reports_produced", "participants_trained", "templates_completed", "avg_hours_saved_per_report"]
    labels = ["Reports", "Participants", "Templates", "Avg Hrs Saved/Report"]
    values = [metrics.get("reports_produced",0),
              metrics.get("participants_trained",0),
              metrics.get("templates_completed",0),
              metrics.get("avg_hours_saved_per_report",0)]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, values)
    ax.set_title("Core KPIs")
    ax.set_ylabel("Value")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ------------- UI ------------- 
st.title("🌍 Sustainably – Impact MVP")
st.caption("Upload CSV/Excel → Map a few columns → Generate IMF (Excel) + Impact Report (.docx)")

with st.sidebar:
    st.header("1) Upload your data")
    upload = st.file_uploader("CSV or Excel", type=["csv","xlsx","xls"])
    data_obj, kind = read_data(upload)

    sheet_options = []
    selected_sheet = None
    if kind == "XLSX" and isinstance(data_obj, pd.ExcelFile):
        sheet_options = data_obj.sheet_names
        selected_sheet = st.selectbox("Sheet", sheet_options, index=0)

    st.header("2) Choose frameworks")
    frameworks = st.multiselect("Align to", ["SDG", "IRIS+", "CIW"], default=["SDG"])

st.write("### Data Preview")
df = load_sheet(data_obj, selected_sheet) if upload else None
if df is not None and not isinstance(df, pd.DataFrame):
    df = load_sheet(data_obj, selected_sheet)
if isinstance(df, pd.DataFrame):
    df = clean_df(df)
    st.dataframe(df.head(20), use_container_width=True)
else:
    st.info("Upload a CSV/XLSX to begin. A sample file is included in the repo if needed.")

# Column mapping
st.write("### 3) Map your columns")
mapping_cols = {}
if isinstance(df, pd.DataFrame):
    colnames = ["—"] + list(df.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        mapping_cols["report_id"] = st.selectbox("Report ID column (unique per report)", colnames, index=colnames.index("report_id") if "report_id" in colnames else 0)
        mapping_cols["participant_id"] = st.selectbox("Participant ID column", colnames, index=colnames.index("participant_id") if "participant_id" in colnames else 0)
    with c2:
        mapping_cols["satisfaction"] = st.selectbox("Satisfaction (1–5) column", colnames, index=colnames.index("satisfaction") if "satisfaction" in colnames else 0)
        mapping_cols["feedback_text"] = st.selectbox("Feedback text column", colnames, index=colnames.index("feedback_text") if "feedback_text" in colnames else 0)
    with c3:
        mapping_cols["program"] = st.selectbox("Program/Project column", colnames, index=colnames.index("program") if "program" in colnames else 0)

    # Normalize "—"
    for k, v in mapping_cols.items():
        if v == "—":
            mapping_cols[k] = None

st.write("### 4) Parameters")
baseline_hours = st.slider("Baseline manual hours per report", 1.0, 12.0, 6.0, 0.5)
automated_hours = st.slider("Automated hours per report (with Sustainably)", 0.0, 8.0, 2.0, 0.5)
hourly_rate = st.slider("Blended hourly rate ($/hr)", 10, 150, 35, 5)

params = {"baseline_hours": baseline_hours, "automated_hours": automated_hours, "hourly_rate": float(hourly_rate)}

# Compute KPIs
metrics = {}
if isinstance(df, pd.DataFrame):
    metrics = compute_kpis(df, mapping_cols, params)
    st.write("### KPIs")
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Reports Produced", metrics.get("reports_produced", 0))
    kpi_cols[1].metric("Participants Trained", metrics.get("participants_trained", 0))
    kpi_cols[2].metric("Avg Hrs Saved/Report", metrics.get("avg_hours_saved_per_report", 0))
    kpi_cols[3].metric("Money Saved ($)", metrics.get("money_saved", 0))

    # Chart
    chart_buf = make_kpi_bar(metrics)
    st.image(chart_buf, caption="Core KPIs", use_column_width=False)

st.write("---")
st.write("### 5) Generate Outputs")

col_a, col_b, col_c = st.columns([1,1,1])

with col_a:
    if st.button("📊 Download IMF (Excel)", use_container_width=True, type="primary"):
        imf_bytes = build_imf_excel(DEFAULT_OUTPUTS)
        st.download_button("Save IMF.xlsx", data=imf_bytes, file_name="Sustainably_IMF.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with col_b:
    if st.button("📝 Download Impact Report (.docx)", use_container_width=True):
        kpi_chart = make_kpi_bar(metrics) if metrics else None
        doc_bytes = build_docx_report(metrics, frameworks, DEFAULT_OUTPUTS, kpi_chart=kpi_chart)
        st.download_button("Save Impact_Report.docx", data=doc_bytes, file_name="Sustainably_Impact_Report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

with col_c:
    if st.button("☁️ Save both to S3 (optional)", use_container_width=True):
        imf_bytes = build_imf_excel(DEFAULT_OUTPUTS)
        doc_bytes = build_docx_report(metrics, frameworks, DEFAULT_OUTPUTS, kpi_chart=make_kpi_bar(metrics) if metrics else None)
        ok1, loc1 = maybe_upload_s3(imf_bytes, f"reports/Sustainably_IMF_{datetime.utcnow().isoformat()}.xlsx")
        ok2, loc2 = maybe_upload_s3(doc_bytes, f"reports/Sustainably_Impact_Report_{datetime.utcnow().isoformat()}.docx")
        st.success(f"IMF upload: {ok1} {loc1}")
        st.success(f"DOCX upload: {ok2} {loc2}")

st.write("---")
st.caption("Tip: Set OPENAI_API_KEY for richer narratives. Without it, the executive summary uses a smart fallback.")


