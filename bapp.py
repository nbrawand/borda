# app.py
import io
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

st.set_page_config(page_title="Borda Count Ranker", layout="wide")

st.title("Borda Count Ranker")
st.caption("""Upload a CSV where **rows are dimensions** and **columns are items** (cells are scores 1–5).

## Understanding Borda Ranking

### What is Borda Ranking?
Borda ranking is a **consensus-based method** for combining preferences across multiple criteria.  
Instead of just averaging raw scores, it looks at the **relative standing of each item within each dimension**:

- For each dimension, items are ranked from best to worst.  
- The top item gets the most points, the next gets slightly fewer, and so on.  
- Ties share the average of the points they would have received.  
- Points are then **summed across dimensions**, and the item with the highest total ranks #1 overall.  

---

### Why use Borda instead of simple aggregation?
- **Removes scale bias** – Different dimensions may use the 1–5 scale differently (some raters are stricter or looser). Borda focuses on *relative order*, not raw numbers.  
- **Balances outliers** – A single extreme score can dominate an average, but in Borda an item must rank well consistently.  
- **Handles ties fairly** – Equal performance within a dimension gives equal credit.  
- **Favors consensus** – Borda highlights items that are strong across the board, rather than polarizing ones that excel in some areas but fail in others.  

---

✅ **In short:** Borda ranking rewards **consistent, well-rounded performance** across all dimensions, while simple averaging can be skewed by scale differences, outliers, or uneven scoring.


""")

with st.expander("CSV format & example"):
    st.markdown(
        """
**Expected layout**

- Rows = dimensions (e.g., Dim 1, Dim 2, …)  
- Columns = items (e.g., Item A, Item B, …)  
- Cells = numeric scores (1–5; higher is better)

**Example Table**
    Item A,Item B,Item C
Dim1  1,     5,     3
Dim2  4,     5,     2
Dim3  2,     3,     1
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
    non_numeric = df_num.isna() & df.notna()
    if non_numeric.any().any():
        bad = np.where(non_numeric)
        bad_cells = [(df.index[i], df.columns[j]) for i, j in zip(*bad)]
        return None, bad_cells, None, None, None

    m = df_num.shape[1]  # number of items
    # Per-dimension ranks (1 = best); ties get average rank
    ranks = df_num.rank(axis=1, ascending=False, method="average")

    # Borda points per dimension: (m - rank), so best gets m-1, lowest 0; ties share average
    points = m - ranks

    # Totals
    total_points = points.sum(axis=0).to_frame(name="total_points")
    max_per_dim = (m - 1)
    max_total = max_per_dim * df_num.shape[0]
    total_points["normalized"] = (total_points["total_points"] / max_total).round(4)

    # Aggregate rank diagnostics
    avg_rank = ranks.mean(axis=0).rename("avg_rank")
    std_rank = ranks.std(axis=0, ddof=0).rename("rank_std")

    # Count (fractional) wins per dimension: give 1/k to each tied highest
    # Find per-dim highest raw score, then split credit among ties
    dim_max = df_num.max(axis=1)
    wins_fractional = []
    for d in df_num.index:
        row = df_num.loc[d]
        winners = row[row == dim_max.loc[d]].index
        credit = 1.0 / len(winners)
        wins_fractional.append(pd.Series({w: credit for w in winners}, index=df_num.columns).fillna(0.0))
    wins_fractional = pd.DataFrame(wins_fractional, index=df_num.index).sum(axis=0).rename("wins")

    summary = pd.concat([total_points, avg_rank, std_rank, wins_fractional], axis=1).sort_values("total_points", ascending=False)
    summary["rank"] = np.arange(1, len(summary) + 1)

    return summary, None, points, ranks, df_num

def _buf_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf.getvalue()

def explain_item(name: str, row: pd.Series, n_dims: int, m_items: int) -> str:
    # Construct a concise narrative
    normalized_pct = 100 * row["normalized"]
    avg_r = row["avg_rank"]
    std_r = row["rank_std"]
    wins = row["wins"]

    lines = []
    lines.append(f"**Overall**: Rank **{int(row['rank'])}** out of **{m_items}** items.")
    lines.append(f"**Borda points**: {row['total_points']:.2f} (≈ **{normalized_pct:.1f}%** of the maximum possible).")
    lines.append(f"**Average per-dimension rank**: {avg_r:.2f} (lower is better); **consistency** (rank std): {std_r:.2f}.")
    lines.append(f"**Dimension wins** (fractional for ties): {wins:.2f} out of {n_dims}.")
    # Heuristics
    if normalized_pct >= 85:
        lines.append("Interpretation: This item performs **consistently near the top** across dimensions.")
    elif normalized_pct >= 65:
        lines.append("Interpretation: **Strong overall** with some **dimension trade-offs**.")
    elif normalized_pct >= 45:
        lines.append("Interpretation: **Middle-of-the-pack**; consider whether certain dimensions matter more.")
    else:
        lines.append("Interpretation: **Generally weaker**; succeeds mainly on a subset of dimensions.")
    return "<br>".join(lines)

def plot_heatmap_scores(df_num: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(min(12, 1.2 * df_num.shape[1] + 4), min(12, 0.45 * df_num.shape[0] + 3)))
    im = ax.imshow(df_num.values, aspect="auto")
    ax.set_xticks(np.arange(df_num.shape[1]))
    ax.set_xticklabels(df_num.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(df_num.shape[0]))
    ax.set_yticklabels(df_num.index)
    ax.set_title("Raw Score Heatmap (dimensions × items)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")
    fig.tight_layout()
    return fig, ax

def plot_borda_bars(summary: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    ordered = summary.sort_values("total_points", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(ordered) + 2))
    ax.barh(ordered.index, ordered["total_points"])
    ax.set_xlabel("Total Borda Points")
    ax.set_title("Borda Totals by Item")
    for i, v in enumerate(ordered["total_points"].values):
        ax.text(v, i, f" {v:.1f}", va="center")
    fig.tight_layout()
    return fig, ax

def plot_rank_profile(ranks: pd.DataFrame, item: str) -> Tuple[plt.Figure, plt.Axes]:
    # Show the per-dimension rank for one item
    r = ranks[item]
    fig, ax = plt.subplots(figsize=(8, 3 + 0.2 * len(r)))
    ax.plot(np.arange(len(r)), r.values, marker="o")
    ax.set_xticks(np.arange(len(r)))
    ax.set_xticklabels(r.index, rotation=45, ha="right")
    ax.set_ylabel("Rank (1=best)")
    ax.set_ylim(0.5, ranks.shape[1] + 0.5)  # visual padding
    ax.set_title(f"Per-Dimension Rank Profile: {item}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax

if f is not None:
    try:
        df = read_csv(f, delim, has_header, index_col_0)
        st.subheader("Preview")
        st.dataframe(df, use_container_width=True)

        with st.spinner("Computing Borda rankings…"):
            summary, bad_cells, per_dim_points, ranks, df_num = borda_count(df)

        if bad_cells is not None:
            st.error(
                "Some cells are non-numeric. Please fix these and re-upload:\n\n"
                + "\n".join([f"- Dimension '{r}', Item '{c}'" for r, c in bad_cells])
            )
        else:
            st.subheader("Final Ranking (Borda)")
            st.dataframe(summary, use_container_width=True)

            # Download final ranking
            csv_buf = io.StringIO()
            summary.to_csv(csv_buf)
            st.download_button(
                label="⬇️ Download ranking (CSV)",
                data=csv_buf.getvalue(),
                file_name="borda_ranking.csv",
                mime="text/csv",
            )

            # --- Interpretations per item ---
            st.markdown("### Item Interpretations")
            n_dims, m_items = df_num.shape[0], df_num.shape[1]
            for item in summary.index:
                with st.expander(f"Interpretation: {item} (Rank {int(summary.loc[item, 'rank'])})", expanded=False):
                    html = explain_item(item, summary.loc[item], n_dims, m_items)
                    st.markdown(html, unsafe_allow_html=True)

                    # Optional per-item rank profile + download
                    fig_r, _ = plot_rank_profile(ranks, item)
                    st.pyplot(fig_r)
                    png_r = _buf_png(fig_r)
                    st.download_button(
                        label=f"⬇️ Download rank profile for {item} (PNG)",
                        data=png_r,
                        file_name=f"{item}_rank_profile.png",
                        mime="image/png",
                    )
                    plt.close(fig_r)

            # --- Visualizations ---
            st.markdown("### Visualizations")
            tab1, tab2 = st.tabs(["Heatmap: Raw Scores", "Bar Chart: Borda Totals"])

            with tab1:
                fig_hm, _ = plot_heatmap_scores(df_num)
                st.pyplot(fig_hm)
                png_hm = _buf_png(fig_hm)
                st.download_button(
                    label="⬇️ Download heatmap (PNG)",
                    data=png_hm,
                    file_name="scores_heatmap.png",
                    mime="image/png",
                )
                plt.close(fig_hm)

            with tab2:
                fig_bar, _ = plot_borda_bars(summary)
                st.pyplot(fig_bar)
                png_bar = _buf_png(fig_bar)
                st.download_button(
                    label="⬇️ Download bar chart (PNG)",
                    data=png_bar,
                    file_name="borda_totals.png",
                    mime="image/png",
                )
                plt.close(fig_bar)

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
**Method notes**

- For *m* items, within each dimension the highest score receives **m−1** points, next gets **m−2**, … lowest gets **0**.  
- Ties share the **average** of the tied positions’ points (via average ranks).  
- Final score is the **sum across dimensions**; we also show a 0–1 **normalized** score.  
- “Dimension wins” give each top-tied item **1/k** credit when *k* items tie for the highest raw score within a dimension.
                """
            )
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")

st.sidebar.header("Tips")
st.sidebar.markdown(
    """
- If some dimensions are more important, add **weights** (easy to extend: weight either raw scores before ranking or Borda points after).
- If “lower is better” for any dimension, invert that column’s values for ranking on that dimension.
- Use the **rank profile** plot to spot items that win on a few dimensions but lag elsewhere (high variance).
"""
)

st.sidebar.write("Made with ❤️ Streamlit")

