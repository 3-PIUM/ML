# app/color_api.py
from fastapi import APIRouter, UploadFile, File
from app.model_loader import get_season
import shutil

router = APIRouter()

@router.post("/predict-season")
async def predict_season(file: UploadFile = File(...)):
    file_path = "temp.jpg"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result_idx = get_season(file_path)
    labels = ['봄웜', '여름쿨', '가을웜', '겨울쿨']
    
    return {"season": labels[result_idx]}
