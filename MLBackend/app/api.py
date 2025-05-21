from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.pkl")

# ✅ 여기에 입력 데이터 스키마를 정의
class IrisRequest(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

# ✅ 여기에 라우터 수정
@app.post("/predict")
async def predict(data: IrisRequest):
    df = pd.DataFrame([{
        "sepal length (cm)": data.sepal_length_cm,
        "sepal width (cm)": data.sepal_width_cm,
        "petal length (cm)": data.petal_length_cm,
        "petal width (cm)": data.petal_width_cm,
    }])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}
