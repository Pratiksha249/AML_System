import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import run_aml_pipeline

st.title("Agentic AI AML Investigation System")

st.write("Upload banking transaction dataset for AML analysis")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    # Read uploaded dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if st.button("Run AML Analysis"):

        # Run AML pipeline
        result = run_aml_pipeline(df)

        # -------------------------------
        # 1. BAR CHART
        # -------------------------------
        st.subheader("Risk Distribution (Bar Chart)")

        risk_counts = result["Risk_Level"].value_counts()

        st.bar_chart(risk_counts)

        # -------------------------------
        # 2. PIE CHART
        # -------------------------------
        st.subheader("Risk Distribution (Pie Chart)")

        fig, ax = plt.subplots()
        ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%')
        ax.set_title("Risk Distribution")

        st.pyplot(fig)

        # -------------------------------
        # 3. FRAUD VS RISK TABLE
        # -------------------------------
        st.subheader("Fraud vs Risk Comparison")

        comparison = pd.crosstab(result["Risk_Level"], result["Is_Laundering"])

        st.dataframe(comparison)

        # -------------------------------
        # 4. TRANSACTION AMOUNT GRAPH
        # -------------------------------
        st.subheader("Transaction Amount Trend")

        st.line_chart(result["Amount_Paid"].head(200))

        # -------------------------------
        # 5. SUSPICIOUS TRANSACTIONS
        # -------------------------------
        st.subheader("Suspicious Transactions")

        suspicious = result[result["Risk_Level"].isin(["Medium", "High"])]

        st.write("Total Suspicious Transactions:", len(suspicious))

        st.dataframe(suspicious.head(50))

        # -------------------------------
        # 6. DOWNLOAD BUTTON
        # -------------------------------
        csv = suspicious.to_csv(index=False)

        st.download_button(
            label="Download Suspicious Transactions",
            data=csv,
            file_name="aml_flagged_transactions.csv",
            mime="text/csv"
        )