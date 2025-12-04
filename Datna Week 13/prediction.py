import joblib
import streamlit as st # Import Streamlit untuk fungsi caching

# Menggunakan @st.cache_resource untuk memuat objek model HANYA SEKALI
# Ini sangat penting untuk performa yang cepat saat di-deploy
@st.cache_resource
def load_model():
    # PERBAIKAN: Mengubah "knn_model_sav" menjadi "knn_model.sav"
    # Memuat file model yang sudah disimpan
    model = joblib.load("knn_model.sav") 
    return model

def predict(data):
    # Panggil fungsi yang sudah di-cache untuk mendapatkan objek model (cepat)
    clf = load_model()
    return clf.predict(data)