import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Arquetipos de jugadores", page_icon="🎮")

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_clustering_jugadores.pkl")

m = cargar_modelo()

ESTRATEGIAS = {
    "Hardcore frecuentes": "Contenido avanzado, eventos competitivos y programa de embajadores.",
    "Sesiones largas esporádicas": "Recordatorios y recompensas semanales; contenido pensado para sesiones largas.",
    "Alta dedicación acumulada": "Misiones cortas y contenido episódico; pase de temporada o suscripción.",
    "Casuales": "Onboarding, retención temprana y ofertas de bajo costo.",
    "Compradores": "Ofertas y bundles personalizados, beneficios VIP y compra sin fricción.",
}

st.title("🎮 Segmentación de jugadores")
st.write("Ingresa los datos de un jugador para conocer su arquetipo y la estrategia sugerida.")

with st.sidebar:
    st.header("Datos del jugador")
    horas = st.slider("Horas jugadas (PlayTimeHours)", 0.0, 24.0, 12.0, 0.1)
    sesiones = st.slider("Sesiones por semana", 0, 19, 9)
    duracion = st.slider("Duración media de sesión (min)", 10, 179, 95)
    nivel = st.slider("Nivel del jugador", 1, 99, 50)
    logros = st.slider("Logros desbloqueados", 0, 49, 25)
    compra = st.radio("¿Realiza compras dentro del juego?", ["No", "Sí"], horizontal=True)

nuevo = pd.DataFrame([{
    "PlayTimeHours": horas,
    "SessionsPerWeek": sesiones,
    "AvgSessionDurationMinutes": duracion,
    "PlayerLevel": nivel,
    "AchievementsUnlocked": logros,
    "InGamePurchases": 1 if compra == "Sí" else 0,
}])[m["features"]]

cluster = int(m["kmeans"].predict(m["scaler"].transform(nuevo))[0])
arquetipo = m["names"][cluster]

st.subheader(f"Arquetipo: {arquetipo}")
st.info(ESTRATEGIAS.get(arquetipo, ""))
st.caption("Modelo K-Means (k=5) entrenado sobre 40.034 jugadores. Segmentos orientativos: la estructura de los datos es débil.")
