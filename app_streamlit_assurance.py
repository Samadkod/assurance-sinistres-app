
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Sinistres Assurance Auto", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("DB_TELEMATICS_PROPRE_I.csv")
    return df

df = load_data()

st.title("🚗 Pilotage des Sinistres d'Assurance Auto")
st.markdown("Une application pour explorer les données et prédire les montants de sinistres.")

# Affichage d’un échantillon
st.subheader("Aperçu des données")
st.dataframe(df.head())

# Filtrage utilisateur
st.sidebar.header("🎯 Filtres")
zone = st.sidebar.selectbox("Zone", df["Zone"].dropna().unique())
categorie = st.sidebar.selectbox("Usage du véhicule", df["Car_use"].dropna().unique())
filtered_df = df[(df["Zone"] == zone) & (df["Car_use"] == categorie)]

st.subheader("📊 Données filtrées")
st.dataframe(filtered_df)

# Modélisation
st.subheader("📈 Prédiction du montant des sinistres")

# Préparation
features = ['Age', 'Bonus_malus', 'Veh_value', 'Exposure']
target = 'Claim_amount'

X = filtered_df[features]
y = filtered_df[target]

# Séparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.markdown(f"**📌 RMSE sur l’échantillon test : {rmse:.2f}**")
