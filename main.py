from fastapi import FastAPI
from pydantic import BaseModel
from utils import download_model
import torch

app = FastAPI()

# Step 1: Download and load the model
download_model()  # Downloads from Google Drive to model/translator_model.pt
model = torch.load("model/translator_model.pt", map_location=torch.device('cpu'))
model.eval()

# Step 2: Define request body
class TranslationRequest(BaseModel):
    text: str

# Step 3: Define route
@app.post("/translate")
def translate_text(request: TranslationRequest):
    input_text = request.text

    # Make prediction with your model
    # 👇 Replace this line with your model's actual prediction method
    output = model.translate(input_text)  # You may need to adapt this

    return {"translation": output}
