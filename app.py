
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Page Config
st.set_page_config(page_title="Equipment Dashboard", layout="wide")
st.title("🛠 Equipment Dashboard")

# Sidebar for Feature Selection
feature = st.sidebar.radio("Select a Feature", ["Data Analysis", "Predictive Maintenance"])

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    if feature == "Data Analysis":
        st.header("📊 Equipment Failure Analysis")
        
        # Sidebar Filters
        st.sidebar.header("Filters")
        model_selected = st.sidebar.multiselect("Select Model(s)", df["model_number"].unique())
        part_selected = st.sidebar.multiselect("Select Part(s)", df["Material Name"].unique())

        if model_selected:
            df = df[df["model_number"].isin(model_selected)]
        if part_selected:
            df = df[df["Material Name"].isin(part_selected)]

        if st.button("Run Equipment Analysis"):
            failure_by_model = df.groupby("model_number").size().reset_index(name="Failure Count")
            failure_by_part = df.groupby("Material Name").size().reset_index(name="Failure Count")
            total_failures = len(df)

            failure_by_model["Failure Rate"] = failure_by_model["Failure Count"] / total_failures
            failure_by_part["Failure Rate"] = failure_by_part["Failure Count"] / total_failures

            st.write("### Top Models by Failure Rate")
            st.dataframe(failure_by_model.sort_values(by="Failure Rate", ascending=False).head(5))
            st.plotly_chart(px.pie(failure_by_model.head(5), names='model_number', values='Failure Count', title='Top Models by Failure Rate'))
            
            st.write("### Top Parts by Failure Rate")
            st.dataframe(failure_by_part.sort_values(by="Failure Rate", ascending=False).head(5))
            st.plotly_chart(px.pie(failure_by_part.head(5), names='Material Name', values='Failure Count', title='Top Parts by Failure Rate'))
            
            # Root Cause Analysis (Apriori Algorithm)
            st.subheader("🔍 Root Cause Analysis")
            df_apriori = df.groupby(["equipment_id", "Material Name"])["Material Name"].count().unstack().fillna(0)
            df_apriori[df_apriori > 0] = 1
            frequent_itemsets = apriori(df_apriori, min_support=0.01, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

            if not rules.empty:
                insights = [f"- If **{', '.join(row['antecedents'])}** is present, there's a **{row['confidence']*100:.2f}%** probability that **{', '.join(row['consequents'])}** will also be required. (Lift: {row['lift']:.2f})" for _, row in rules.sort_values(by="lift", ascending=False).head(5).iterrows()]
                st.write("\n".join(insights))
            else:
                st.write("No strong associations found.")
    
    elif feature == "Predictive Maintenance":
        st.header("🔧 Predictive Maintenance Analysis")
        
        df["created_at"] = pd.to_datetime(df["created_at"], format="%d-%m-%Y")
        df = df.sort_values(by=["equipment_id", "created_at"])
        df["equipment_id"] = df["equipment_id"].astype(str)
        df["Next Failure Date"] = df.groupby("equipment_id")["created_at"].shift(-1)
        df["duration"] = (df["Next Failure Date"] - df["created_at"]).dt.days
        df.drop(columns=["Next Failure Date"], inplace=True)
        df = df.assign(duration=df["duration"].fillna(df["duration"].median()))
        df["event_observed"] = 1
        df.loc[df.sample(frac=0.2, random_state=42).index, "event_observed"] = 0
        df_original = df.copy()

        features_to_encode = ['model_number', 'Material Name', 'original_fault_code', 'Material Type', 'Material Group']
        df = pd.get_dummies(df, columns=features_to_encode, drop_first=True)

        X = df.drop(columns=["equipment_id", "created_at", "closed_at", "duration", "event_observed"])
        y = Surv.from_dataframe("event_observed", "duration", df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, random_state=42)
        rsf.fit(X_train, y_train)

        df["Predicted Time to Failure"] = rsf.predict(X)
        df["Failure Risk Score"] = 1 / (df["Predicted Time to Failure"] + 1)

        user_equipment_id = st.text_input("Enter Equipment ID:")
        if user_equipment_id:
            user_equipment_id = str(user_equipment_id)
            equipment_data = df[df["equipment_id"] == user_equipment_id]

            if equipment_data.empty:
                st.warning(f"No data found for Equipment ID {user_equipment_id}")
            else:
                st.subheader(f"📌 Predicted Maintenance Data for Equipment ID: {user_equipment_id}")
                st.dataframe(equipment_data[["Predicted Time to Failure", "Failure Risk Score"]].head(10))

                material_name_columns = [col for col in df.columns if col.startswith('Material Name')]
                spare_parts = equipment_data.groupby(material_name_columns).agg({"Failure Risk Score": "mean"}).sort_values(by="Failure Risk Score", ascending=False)
                spare_parts.index = spare_parts.index.map(lambda x: next((col.split('_')[1] for col in material_name_columns if x[material_name_columns.index(col)] == 1 and len(col.split('_')) > 1), None))
                spare_parts = spare_parts.dropna()
              
                st.subheader("🛠 Spare Parts Optimization")
                st.dataframe(spare_parts.head(10))
                
                st.subheader("🔧 Most Failed Model Combinations")
                best_combinations = df_original.groupby(["model_number", "equipment_id"]).size().reset_index(name="Failure Count")
                st.dataframe(best_combinations[best_combinations["equipment_id"] == user_equipment_id].sort_values(by="Failure Count", ascending=False).head(10))

                ### Step 6: Visualization ###
                col1, col2 = st.columns(2)

                with col1:
                  st.subheader(f"Predicted Time to Next Failure Distribution")
                  fig, ax = plt.subplots(figsize=(8, 5))
                  sns.histplot(equipment_data["Predicted Time to Failure"], bins=30, kde=True, ax=ax)
                  plt.xlabel("Days")
                  plt.ylabel("Frequency")
                  st.pyplot(fig)

                with col2:
                  st.subheader(f"Top High-Risk Spare Parts")
                  fig, ax = plt.subplots(figsize=(8, 5))
                  sns.barplot(x=spare_parts.index[:10], y=spare_parts["Failure Risk Score"].iloc[:10], ax=ax)
                  plt.xticks(rotation=90)
                  plt.ylabel("Average Failure Risk Score")
                  st.pyplot(fig)
