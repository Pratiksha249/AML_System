import pandas as pd

# Load dataset
df = pd.read_csv("transactions.csv")
print("Original size:", len(df))

# Take sample
df_small = df.sample(20000, random_state=42)

# Save smaller dataset
df_small.to_csv("transactions_small.csv", index=False)

print("Small dataset created successfully!")
# Rename columns
df = df.rename(columns={
    "nameOrig": "Sender_Account",
    "nameDest": "Receiver_Account",
    "amount": "Amount_Paid",
    "step": "Timestamp",
    "isFraud": "Is_Laundering"
})

print("\nRenamed Columns:")
print(df.columns)

# Filter columns
df = df[[
    "Timestamp",
    "Sender_Account",
    "Receiver_Account",
    "Amount_Paid",
    "Is_Laundering"
]]

print("\nFiltered Dataset:")
print(df.head())


# -----------------------------
# Profiling Agent
# -----------------------------
class ProfilingAgent:

    def build_profile(self, df):

        profile = df.groupby("Sender_Account")["Amount_Paid"].agg(
            ["mean", "std", "count"]
        ).reset_index()

        profile.columns = [
            "Sender_Account",
            "Avg_Amount",
            "Std_Amount",
            "Transaction_Count"
        ]

        df = df.merge(profile, on="Sender_Account", how="left")

        return df


# -----------------------------
# TEST THE AGENT
# -----------------------------
profiler = ProfilingAgent()

df_profiled = profiler.build_profile(df)

print("\nProfiled Dataset:")
print(df_profiled.head())


class ReasoningAgent:

    def apply_rules(self, df):

        # replace NaN std values
        df["Std_Amount"] = df["Std_Amount"].fillna(0)

        # Rule 1 — spike detection
        df["Spike_Flag"] = df["Amount_Paid"] > (
            df["Avg_Amount"] + 1.5 * df["Std_Amount"]
        )

        # Rule 2 — extremely high value transaction
        threshold = df["Amount_Paid"].quantile(0.95)

        df["High_Value_Flag"] = df["Amount_Paid"] > threshold

        # Rule 3 — low history suspicious
        df["Low_History_Flag"] = (
            (df["Transaction_Count"] <= 2) &
            (df["Amount_Paid"] > df["Avg_Amount"])
        )

        return df
    
reasoning = ReasoningAgent()

df_reasoned = reasoning.apply_rules(df_profiled)

print("\nReasoning Output:")
print(df_reasoned.head())

class DecisionAgent:

    def classify_risk(self, df):

        df["Risk_Score"] = (
            df["Spike_Flag"].astype(int)
            + df["High_Value_Flag"].astype(int)
            + df["Low_History_Flag"].astype(int)
        )

        def risk(score):

            if score == 0:
                return "Low"

            elif score == 1:
                return "Medium"

            else:
                return "High"

        df["Risk_Level"] = df["Risk_Score"].apply(risk)

        return df
    
decision = DecisionAgent()

df_decision = decision.classify_risk(df_reasoned)

print("\nDecision Output:")
print(df_decision.head())

class ActionAgent:

    def generate_report(self, df):

        flagged = df[df["Risk_Level"] == "High"]

        with open("aml_investigation_report.txt", "w") as f:

            f.write("AML Investigation Report\n")
            f.write("=========================\n\n")

            for _, row in flagged.iterrows():

                f.write(
                    f"Account: {row['Sender_Account']} | "
                    f"Amount: {row['Amount_Paid']} | "
                    f"Risk: {row['Risk_Level']}\n"
                )

        print("\nReport Generated Successfully")

        print("\nRisk Distribution:")
        print(df["Risk_Level"].value_counts())

def main():

    profiler = ProfilingAgent()
    reasoning = ReasoningAgent()
    decision = DecisionAgent()
    action = ActionAgent()

    df1 = profiler.build_profile(df)
    df2 = reasoning.apply_rules(df1)
    df3 = decision.classify_risk(df2)

    action.generate_report(df3)


if __name__ == "__main__":
    main()



    # Rename dataset columns
def run_aml_pipeline(uploaded_df):

    # Rename columns from PaySim dataset
    df = uploaded_df.rename(columns={
        "nameOrig": "Sender_Account",
        "nameDest": "Receiver_Account",
        "amount": "Amount_Paid",
        "step": "Timestamp",
        "isFraud": "Is_Laundering"
    })

    # Keep only required columns
    df = df[[
        "Timestamp",
        "Sender_Account",
        "Receiver_Account",
        "Amount_Paid",
        "Is_Laundering"
    ]]

    profiler = ProfilingAgent()
    reasoning = ReasoningAgent()
    decision = DecisionAgent()

    df1 = profiler.build_profile(df)
    df2 = reasoning.apply_rules(df1)
    df3 = decision.classify_risk(df2)

    return df3



