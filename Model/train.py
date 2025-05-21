# Model/train.py
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import joblib
import os

X, y = load_iris(return_X_y=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X_scaled, y)

joblib.dump(model, "model.pkl")
print("✅ model.pkl 저장 완료")
