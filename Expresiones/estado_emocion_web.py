import streamlit as st
import time
import os

st.set_page_config(page_title="Emoción en Tiempo Real", layout="centered")

st.title("🎭 Emoción en Tiempo Real")
st.markdown("La cámara se está ejecutando por separado. Esta interfaz muestra la emoción más reciente detectada.")

placeholder = st.empty()

emojis = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
    "surprise": "😲",
    "Sin atención": "👀",
}

while True:
    if os.path.exists("estado_emocion.txt"):
        with open("estado_emocion.txt", "r") as f:
            emocion = f.read().strip()
    else:
        emocion = "Sin atención"

    emoji = emojis.get(emocion.lower(), "🤖")

    with placeholder.container():
        st.subheader(f"{emoji} Emoción detectada: **{emocion}**")

    time.sleep(1)
