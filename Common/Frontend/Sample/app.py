import streamlit as st
import pandas as pd


st.set_page_config(page_title="Simple Dividend Dashboard", layout="wide")

st.title("Simple Financial Dashboard (4 Tabs)")
st.caption("Minimal example with metric row, screener table, and expandable details.")

df = pd.read_csv("dividend_sample.csv")


# Macro metrics row (required pattern: st.metric)
avg_yield = df["dividend_yield"].mean()
avg_pe = df["pe_ratio"].mean()
total_market_cap = df["market_cap_bil"].sum()
top_yield_symbol = df.sort_values("dividend_yield", ascending=False).iloc[0]["symbol"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Stocks Tracked", f"{len(df)}")
m2.metric("Avg Dividend Yield", f"{avg_yield:.2f}%")
m3.metric("Avg P/E", f"{avg_pe:.1f}")
m4.metric("Top Yield", top_yield_symbol)


tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Screener", "Company Detail", "Notes"])

with tab1:
    st.subheader("Overview")
    st.write("This is a simple overview tab to mimic a standard financial dashboard layout.")
    st.dataframe(
        df[["symbol", "company", "sector", "price", "dividend_yield"]],
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Screener")
    min_yield = st.slider("Minimum dividend yield (%)", 0.0, 5.0, 1.5, 0.1)
    screener_df = df[df["dividend_yield"] >= min_yield].sort_values(
        "dividend_yield", ascending=False
    )
    st.dataframe(screener_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Company Detail")
    selected_symbol = st.selectbox("Choose a symbol", df["symbol"].tolist())
    row = df[df["symbol"] == selected_symbol].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${row['price']:.2f}")
    c2.metric("Dividend Yield", f"{row['dividend_yield']:.2f}%")
    c3.metric("Last Dividend", f"${row['last_dividend']:.2f}")

    # Detail panel pattern (required: st.expander)
    with st.expander("Show company detail panel"):
        st.write(f"**Company:** {row['company']}")
        st.write(f"**Sector:** {row['sector']}")
        st.write(f"**P/E Ratio:** {row['pe_ratio']:.1f}")
        st.write(f"**Market Cap:** {row['market_cap_bil']}B USD")
        st.write(f"**Payout Frequency:** {row['payout_frequency']}")

with tab4:
    st.subheader("Notes")
    with st.expander("How this template maps to the 4-tab pattern"):
        st.markdown(
            "- **Metric macro row** at the top (`st.metric`).\n"
            "- **Screener table** in tab 2 (`st.dataframe`).\n"
            "- **Detail panel** in tab 3 (`st.expander`).\n"
            "- Four tabs total, matching the common Streamlit financial app structure."
        )
