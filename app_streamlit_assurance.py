import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Pilotage des Sinistres Auto", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("DB_TELEMATICS_PROPRE_I.csv")
    return df

# Chargement des données
df = load_data()

st.title("🚗 Pilotage des Sinistres Auto")
st.markdown("Une application interactive pour explorer les données de sinistres et visualiser les prédictions des montants.")

# Filtrage par usage du véhicule (au lieu de "categorie_vehicule")
if "Car_use" in df.columns:
    selected_usage = st.sidebar.multiselect("Usage du véhicule", options=df["Car_use"].unique(), default=df["Car_use"].unique())
    df = df[df["Car_use"].isin(selected_usage)]
else:
    st.sidebar.warning("⚠️ Colonne 'Car_use' introuvable dans le fichier CSV.")

# Affichage des statistiques descriptives
st.subheader("📊 Aperçu des données")
st.dataframe(df.head())

# Histogramme du nombre de sinistres par tranche d'âge
if "Age_vehicule" in df.columns:
    fig_age = px.histogram(df, x="Age_vehicule", title="Distribution de l'âge des véhicules")
    st.plotly_chart(fig_age, use_container_width=True)

# Visualisation de la gravité moyenne des sinistres
if "Montant_sinistre" in df.columns and "Zone" in df.columns:
    gravite_par_zone = df.groupby("Zone")["Montant_sinistre"].mean().reset_index()
    fig_gravite = px.bar(gravite_par_zone, x="Zone", y="Montant_sinistre", title="Gravité moyenne des sinistres par zone")
    st.plotly_chart(fig_gravite, use_container_width=True)

# 🔮 Simulation de prédiction de sinistre
st.subheader("🔮 Simulation de prédiction")

with st.form("prediction_form"):
    age_veh = st.slider("Âge du véhicule", min_value=0, max_value=30, value=5)
    km_parcourus = st.number_input("Kilomètres parcourus par an", min_value=0, max_value=100000, value=15000)
    zone = st.selectbox("Zone", options=df["Zone"].unique())
    puissance = st.slider("Puissance fiscale", min_value=1, max_value=20, value=6)
    submit = st.form_submit_button("Prédire")

if submit:
    # Simulation d'un modèle prédictif (à remplacer par XGBoost ou autre)
    prediction = 300 + age_veh * 20 + (km_parcourus / 1000) * 15 + puissance * 25
    st.success(f"✅ Montant de sinistre estimé : {round(prediction, 2)} €")

# Footer
st.markdown("---")
st.markdown("📁 Fichier utilisé : `DB_TELEMATICS_PROPRE_I.csv`")
st.markdown("👤 Réalisé par Samadou Kodon")
