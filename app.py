import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ================= UI CONFIG =================
st.set_page_config(page_title="Cybersecurity Dashboard", layout="wide")

st.title("🔐 Cybersecurity Attack Analysis Dashboard")

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("cybersecurity_attacks_data.csv")

st.subheader("📊 Dataset Preview")
st.write(df.head())

# ================= DATA CLEANING =================
df = df.dropna()

# ================= SIDEBAR =================
st.sidebar.title("📌 Dashboard Options")
option = st.sidebar.selectbox("Select View", [
    "Overview",
    "Attack Trends",
    "Country Analysis",
    "Model Training",
    "Prediction"
])

# ================= OVERVIEW =================
if option == "Overview":
    st.subheader("📌 Basic Information")
    st.write(df.describe())

    st.subheader("📊 Attack Type Distribution")
    if "attack_type" in df.columns:
        fig, ax = plt.subplots()
        df["attack_type"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

# ================= ATTACK TRENDS =================
elif option == "Attack Trends":
    st.subheader("📈 Attack Trends Over Time")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        trend = df.groupby(df["date"].dt.date).size()

        fig, ax = plt.subplots()
        trend.plot(ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No 'date' column found")

# ================= COUNTRY ANALYSIS =================
elif option == "Country Analysis":
    st.subheader("🌍 Country-wise Attacks")

    if "country" in df.columns:
        fig, ax = plt.subplots()
        df["country"].value_counts().head(10).plot(kind="bar", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No 'country' column found")

# ================= MODEL TRAINING =================
elif option == "Model Training":
    st.subheader("🤖 Train ML Model")

    target_column = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Model"):
        le = LabelEncoder()

        df_encoded = df.copy()

        for col in df_encoded.select_dtypes(include=["object"]).columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])

        X = df_encoded.drop(target_column, axis=1)
        y = df_encoded[target_column]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.success(f"✅ Model Accuracy: {acc:.2f}")

        joblib.dump(model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")

        st.success("💾 Model Saved as model.pkl")

# ================= PREDICTION =================
elif option == "Prediction":
    st.subheader("🔮 Make Prediction")

    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")

        input_data = []

        for col in df.columns[:-1]:
            val = st.number_input(f"Enter {col}", value=0.0)
            input_data.append(val)

        if st.button("Predict"):
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)

            st.success(f"🎯 Prediction: {prediction[0]}")

    except:
        st.error("⚠️ Train the model first!")

# ================= FOOTER =================
st.markdown("---")
st.markdown("🚀 Developed as Real-Time Data Science Project")