# app.py
import io
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Borda Count Ranker", layout="wide")

st.title("Borda Count Ranker")
st.caption("Upload a CSV where **rows are dimensions** and **columns are items** (cells are scores 1–5).")

with st.expander("CSV format & example"):
    st.markdown(
        """
**Expected layout**

- Rows = dimensions (e.g., Dim 1, Dim 2, …)  
- Columns = items (e.g., Item A, Item B, …)  
- Cells = numeric scores (1–5; higher is better)

**Example CSV**
,Item A,Item B,Item C
Dim 1,5,3,4
Dim 2,2,4,4
Dim 3,3,5,1
"""
    )

f = st.file_uploader("Upload CSV", type=["csv"])
col_opts = st.columns(3)
with col_opts[0]:
    delim = st.selectbox("Delimiter", [",", ";", "\t", "|"], index=0)
with col_opts[1]:
    has_header = st.selectbox("Header row present?", ["Yes", "No"], index=0)
with col_opts[2]:
    index_col_0 = st.selectbox("First column is dimension names?", ["Yes", "No"], index=0)

def read_csv(file, delimiter, header_flag, index_flag):
    header = 0 if header_flag == "Yes" else None
    idx = 0 if index_flag == "Yes" else None
    df = pd.read_csv(file, delimiter=delimiter, header=header, index_col=idx)
    # If no header, auto-generate item names
    if header is None:
        df.columns = [f"Item {i+1}" for i in range(df.shape[1])]
    # If no index, auto-generate dimension names
    if idx is None:
        df.index = [f"Dim {i+1}" for i in range(df.shape[0])]
    return df

@st.cache_data(show_spinner=False)
def borda_count(df: pd.DataFrame):
    # Ensure numeric
    df_num = df.apply(pd.to_numeric, errors="coerce")
    # Report any non-numeric cells
    non_numeric = df_num.isna() & df.notna()
    if non_numeric.any().any():
        bad = np.where(non_numeric)
        bad_cells = [(df.index[i], df.columns[j]) for i, j in zip(*bad)]
        return None, bad_cells, None

    m = df_num.shape[1]  # number of items
    # Rank within each dimension: highest score -> rank 1
    ranks = df_num.rank(axis=1, ascending=False, method="average")
    # Borda points: (m - rank). Rank=1 -> m-1 points. Ties share average points.
    points = m - ranks
    # Sum across dimensions
    total_points = points.sum(axis=0).to_frame(name="total_points")
    # Optional: normalized score 0..1 relative to max possible
    max_per_dim = (m - 1)  # max points per dimension
    max_total = max_per_dim * df_num.shape[0]
    total_points["normalized"] = (total_points["total_points"] / max_total).round(4)

    # Final ranking
    total_points = total_points.sort_values("total_points", ascending=False)
    total_points["rank"] = np.arange(1, len(total_points) + 1)

    return total_points, None, points

if f is not None:
    try:
        df = read_csv(f, delim, has_header, index_col_0)
        st.subheader("Preview")
        st.dataframe(df, use_container_width=True)

        with st.spinner("Computing Borda rankings…"):
            results, bad_cells, per_dim_points = borda_count(df)

        if bad_cells is not None:
            st.error(
                "Some cells are non-numeric. Please fix these and re-upload:\n\n"
                + "\n".join([f"- Dimension '{r}', Item '{c}'" for r, c in bad_cells])
            )
        else:
            st.subheader("Final Ranking (Borda)")
            st.dataframe(results, use_container_width=True)

            # Download final ranking
            csv_buf = io.StringIO()
            results.to_csv(csv_buf)
            st.download_button(
                label="⬇️ Download ranking (CSV)",
                data=csv_buf.getvalue(),
                file_name="borda_ranking.csv",
                mime="text/csv",
            )

            with st.expander("Per-dimension Borda points (diagnostics)"):
                st.caption("Points per dimension per item (higher is better).")
                st.dataframe(per_dim_points, use_container_width=True)
                diag_buf = io.StringIO()
                per_dim_points.to_csv(diag_buf)
                st.download_button(
                    label="⬇️ Download per-dimension points (CSV)",
                    data=diag_buf.getvalue(),
                    file_name="borda_points_by_dimension.csv",
                    mime="text/csv",
                )

            st.markdown(
                """
**Notes on method**

- For *m* items, within each dimension the highest score receives **m−1** points, next gets **m−2**, … lowest gets **0**.  
- Ties share the **average** of the tied positions’ points (via average ranks).  
- Final score is the **sum across all dimensions**; we also show a 0–1 **normalized** score (divide by the maximum possible).  
                """
            )
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")

st.sidebar.header("Tips")
st.sidebar.markdown(
    """
- Normalize your scoring rubric across dimensions if different raters used different scales.
- If some dimensions are more important, consider weighting (not included here, but easy to add).
- Borda reduces sensitivity to raw scale differences by using **ranks per dimension**.
"""
)

st.sidebar.write("Made with ❤️ Streamlit")

