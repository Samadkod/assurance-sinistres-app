import streamlit as st
import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="🚗 Sinistres Auto", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("DB_TELEMATICS_PROPRE_I.csv")

df = load_data()

st.title("🚗 Pilotage des Sinistres Auto - Projet Assurance")
st.markdown("Application interactive pour explorer les données de sinistres et visualiser les prédictions des montants.")

# Afficher les colonnes disponibles
st.write("Colonnes disponibles dans les données :", df.columns.tolist())

# Vérification des colonnes clés
required_columns = ["Car_use", "Age", "Zone", "Bonus_malus", "Veh_value", "Exposure", "Claim_amount"]
missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    st.error(f"Colonnes manquantes : {missing_cols}")
    st.stop()

# Filtrage utilisateur
with st.form("filters_form"):
    selected_use = st.selectbox("Usage du véhicule", options=df["Car_use"].unique())
    zone = st.selectbox("Zone géographique", options=df["Zone"].unique())
    valider = st.form_submit_button("Valider")

if valider:
    filtered_df = df[(df["Car_use"] == selected_use) & (df["Zone"] == zone)]

    st.subheader("🔍 Données filtrées")
    st.dataframe(filtered_df.head())

    st.subheader("📊 Analyse descriptive")
    fig1 = px.histogram(filtered_df, x="Veh_value", title="Répartition de la valeur des véhicules")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📈 Prédiction du montant des sinistres")

    X = filtered_df[["Age", "Bonus_malus", "Veh_value", "Exposure"]]
    y = filtered_df["Claim_amount"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    df_results = pd.DataFrame({"Réel": y_test, "Prédit": y_pred})
    st.write(df_results.head())

    fig2 = px.scatter(df_results, x="Réel", y="Prédit", title="Réel vs Prédit")
    st.plotly_chart(fig2, use_container_width=True)