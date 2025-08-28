# Sustainably – Impact MVP

**Upload CSV/Excel → Map → Generate**

This Streamlit MVP demonstrates Sustainably’s core value: turning messy spreadsheets into an investor-ready **Impact Report (.docx)** and a structured **Impact Measurement Framework (IMF.xlsx)** aligned to recognized frameworks (SDGs/IRIS+/CIW).

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

Optional environment variables (use a `.env` file):
```
OPENAI_API_KEY=your_key_here   # enables richer executive summary
OPENAI_MODEL=gpt-4o-mini       # or gpt-4, gpt-3.5-turbo (depending on account)
AWS_S3_BUCKET=your-bucket
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
```

A sample dataset is included: `nonprofit_sample_data.xlsx` (sheet: `data`).

## Flow
1. Upload CSV/XLSX → Preview.
2. Map a few columns (report_id, participant_id, satisfaction, feedback, program).
3. Set parameters (baseline vs automated hours; hourly rate).
4. Generate downloads:
   - `Sustainably_IMF.xlsx` (first 5 columns for 9 outputs)
   - `Sustainably_Impact_Report.docx` (KPIs, narrative, charts, framework tags)

## Notes
- Works entirely on files; no database.
- LLM is optional; clean fallback narrative if no API key.
- S3 save is optional; requires bucket + creds.
