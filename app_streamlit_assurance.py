
import streamlit as st
import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="🚗 Pilotage des Sinistres Auto", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("DB_TELEMATICS_PROPRE_I.csv")
    return df

df = load_data()

st.title("🚗 Pilotage des Sinistres Auto - Application Streamlit")
st.markdown("**Une application interactive pour explorer les données de sinistres et visualiser les prédictions.**")

# Affichage des premières lignes
if st.checkbox("Afficher un échantillon des données"):
    st.dataframe(df.head(20))

# Filtres
st.sidebar.header("🎯 Filtres")
selected_zone = st.sidebar.multiselect("Zone", options=df["Zone"].dropna().unique(), default=list(df["Zone"].dropna().unique()))
selected_car_use = st.sidebar.multiselect("Usage véhicule", options=df["Car_use"].dropna().unique(), default=list(df["Car_use"].dropna().unique()))
selected_fuel = st.sidebar.multiselect("Carburant", options=df["Fuel"].dropna().unique(), default=list(df["Fuel"].dropna().unique()))

filtered_df = df[
    df["Zone"].isin(selected_zone) &
    df["Car_use"].isin(selected_car_use) &
    df["Fuel"].isin(selected_fuel)
]

# Visualisation
st.subheader("📊 Analyse des sinistres")
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(px.histogram(filtered_df, x="Veh_value", nbins=30, title="Valeur des véhicules"))

with col2:
    st.plotly_chart(px.box(filtered_df, x="Claim_amount", title="Répartition des montants de sinistres"))

# Modélisation XGBoost
st.subheader("🧠 Prédiction du montant des sinistres (XGBoost)")

features = ["Age", "Zone", "Bonus_malus", "Veh_value", "Exposure"]
target = "Claim_amount"

try:
    X = pd.get_dummies(filtered_df[features])
    y = filtered_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.markdown(f"**RMSE :** {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    st.markdown(f"**R² :** {r2_score(y_test, y_pred):.2f}")

    st.markdown("### 🔍 Exemple de prédictions")
    results_df = pd.DataFrame({"Réel": y_test, "Prédit": y_pred}).reset_index(drop=True)
    st.dataframe(results_df.head(10))

except Exception as e:
    st.error(f"Erreur lors de l'entraînement du modèle : {e}")
