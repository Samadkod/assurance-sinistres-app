# app_streamlit_assurance.py

import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Sinistres Auto - Assurance", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("DB_TELEMATICS_PROPRE_I.csv")
    df = df.rename(columns={
        "Veh_value": "Veh_value",
        "Car_use": "Car_use",
        "Zone": "Zone",
        "Age": "Age",
        "Bonus_malus": "Bonus_malus",
        "Exposure": "Exposure",
        "Claim_amount": "Claim_amount"
    })
    return df

df = load_data()

# Interface utilisateur
st.title("🚗 Pilotage des Sinistres Auto")
st.markdown("Une application interactive pour explorer les données de sinistres et visualiser les prédictions.")

with st.sidebar:
    st.header("🎯 Filtres")
    zone = st.selectbox("Zone", options=df["Zone"].unique())
    car_use = st.selectbox("Utilisation du véhicule", options=df["Car_use"].unique())
    age = st.slider("Âge du conducteur", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].mean()))
    bonus = st.slider("Bonus/Malus", int(df["Bonus_malus"].min()), int(df["Bonus_malus"].max()), int(df["Bonus_malus"].mean()))

# Filtrage des données
df_filtered = df[
    (df["Zone"] == zone) &
    (df["Car_use"] == car_use) &
    (df["Age"] == age) &
    (df["Bonus_malus"] == bonus)
]

st.write(f"Nombre d'observations après filtrage : {df_filtered.shape[0]}")
st.dataframe(df_filtered.head(10))

# Préparation du modèle XGBoost
st.subheader("📊 Modélisation XGBoost")

features = ['Age', 'Zone', 'Bonus_malus', 'Veh_value', 'Exposure']
df_model = df.dropna(subset=features + ['Claim_amount'])
X = pd.get_dummies(df_model[features])
y = df_model["Claim_amount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective="reg:squarederror")
model.fit(X_train, y_train)

# Prédiction et performance
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.metric("RMSE du modèle", f"{rmse:.2f} €")
st.metric("R²", f"{r2:.2%}")

# Prédiction sur filtres
if df_filtered.shape[0] > 0:
    X_input = pd.get_dummies(df_filtered[features])
    X_input = X_input.reindex(columns=X.columns, fill_value=0)
    y_input_pred = model.predict(X_input)

    st.subheader("📈 Prédictions pour les filtres choisis")
    df_filtered = df_filtered.copy()
    df_filtered["Montant prédit (€)"] = y_input_pred
    st.dataframe(df_filtered[features + ["Claim_amount", "Montant prédit (€)"]])
