import joblib

def predict(data):
      clf = joblib.load("knn_model_sav")
  return clf.predict(data)