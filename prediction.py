import joblib
import streamlit as st 

@st.cache_resource 
def load_model():
    model = joblib.load("knn_model.sav") 
    return model

def predict(data):
    clf = load_model()
    return clf.predict(data)
