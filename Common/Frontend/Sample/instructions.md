# IMPORTANT NOTE: THIS IS ONLY A SAMPLE

This MUST NOT be used as representation of what the intended app design should look like.
ALWAYS prefer to use the actual DuckDB contents, user instructions, as a guide.

# Run Instructions

## 1) Install dependencies

```powershell
pip install streamlit pandas
```

## 2) Run the dashboard

From this folder (`d:\WS_Work\IRS_Test_Streamlit`):

```powershell
streamlit run app.py
```

## 3) What you should see

- A top **metric row** using `st.metric()`
- A **four-tab layout**
- A **screener table** using `st.dataframe()`
- **Detail panels** using `st.expander()`
