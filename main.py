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
    from utils import tokenizer  # Make sure you have a tokenizer object in utils.py
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_tokens = model.generate(input_ids, num_beams=10, num_return_sequences=1)
    output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return {"translation": output}
