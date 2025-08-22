from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from predict import predict_tags

app = FastAPI()

# Cho phép gọi API từ browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve file index.html khi mở /
@app.get("/")
def root():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(file_path)

# Data model
class InputText(BaseModel):
    sentence: str

# API POS Tagging
@app.post("/pos-tag")
def pos_tag(data: InputText):
    results = predict_tags(data.sentence)
    return {"tags": [{"word": w, "pos": t} for w, t in results]}
