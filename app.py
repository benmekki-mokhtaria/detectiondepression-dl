
import streamlit as st
import tensorflow as tf
import numpy as np
import re

# Configuration minimale
st.set_page_config(page_title="MentalGuard", layout="centered")
st.title("🧠 MentalGuard AI")
st.write("Système d'analyse émotionnelle")

# Fonction simple
def clean_text(text):
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join(text.split())

# Interface
text_input = st.text_area("Entrez votre texte:", height=120)

if st.button("Analyser", type="primary"):
    if text_input and text_input.strip():
        with st.spinner("Analyse en cours..."):
            try:
                # Charger le modèle
                model = tf.keras.models.load_model("mon_modele_depression_final.h5")
                
                # Charger tokenizer
                import pickle
                with open("tokenizer.pkl", "rb") as f:
                    tokenizer = pickle.load(f)
                
                # Préparation
                cleaned_text = clean_text(text_input)
                sequences = tokenizer.texts_to_sequences([cleaned_text])
                padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100, padding="post")
                
                # Prédiction
                predictions = model.predict(padded, verbose=0)
                class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                # Résultat
                st.success(f"**Niveau {class_idx}** détecté")
                st.info(f"**Confiance: {confidence:.1%}**")
                
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    else:
        st.warning("Veuillez entrer un texte")
