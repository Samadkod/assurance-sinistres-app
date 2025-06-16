
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Titre principal
st.title("🚗 Pilotage des Sinistres Auto - Projet Assurance")
st.markdown("Une application interactive pour explorer les données de sinistres et visualiser les prédictions des montants.")

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("DB_TELEMATICS_PROPRE_I.csv")

df = load_data()

# Sélecteurs pour filtrer les données
st.sidebar.header("🎯 Filtres")
selected_vehicules = st.sidebar.multiselect("Catégorie véhicule", options=df["categorie_vehicule"].unique(), default=df["categorie_vehicule"].unique())
selected_villes = st.sidebar.multiselect("Ville", options=df["ville"].unique(), default=df["ville"].unique())

df_filtered = df[(df["categorie_vehicule"].isin(selected_vehicules)) & (df["ville"].isin(selected_villes))]

# Affichage des données filtrées
st.subheader("🔍 Aperçu des données filtrées")
st.dataframe(df_filtered.head())

# Visualisation du montant moyen des sinistres par ville
st.subheader("📊 Montant moyen des sinistres par ville")
avg_sinistres = df_filtered.groupby("ville")["montant_sinistre"].mean().reset_index()
fig1 = px.bar(avg_sinistres, x="ville", y="montant_sinistre", title="Montant moyen des sinistres par ville", text_auto=".2s")
st.plotly_chart(fig1)

# Carte des sinistres
if "longitude" in df_filtered.columns and "latitude" in df_filtered.columns:
    st.subheader("🗺️ Carte des sinistres")
    st.map(df_filtered[["latitude", "longitude"]])

# Benchmark des modèles
st.subheader("⚙️ Benchmark des Modèles")
benchmark_data = {
    "Modèle": ["Random Forest", "XGBoost", "GLM Tweedie"],
    "RMSE": [191.82, 196.55, 212.43]
}
df_benchmark = pd.DataFrame(benchmark_data)
st.dataframe(df_benchmark)

# Footer
st.markdown("---")
st.markdown("Projet réalisé par **Samadou Kodon** – Portfolio : [https://samadkod.github.io](https://samadkod.github.io)")
