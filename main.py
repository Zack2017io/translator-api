from fastapi import FastAPI
from pydantic import BaseModel
from utils import download_model


app = FastAPI()

# Step 1: Download and load the model
download_model()  # Downloads from Google Drive to model/translator_model.pt
# The model will be loaded using HuggingFace Transformers below, so torch.load is not needed.


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define the model name (replace with your actual model name if different)
model_name = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Define LANG_TOKEN_MAPPING or import it from utils if it exists there
LANG_TOKEN_MAPPING = {
    "<en>": "English",
    "<sw>": "Swahili",
    # Add more language tokens as needed
}
special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
tokenizer.add_special_tokens(special_tokens_dict)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))


# Step 2: Define request body
class TranslationRequest(BaseModel):
    text: str

# Step 3: Define route
@app.post("/translate")
def translate_text(request: TranslationRequest):
    input_text = f"translate English to Swahili: {request.text}"

    # Make prediction with your model
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_tokens = model.generate(input_ids, num_beams=10, num_return_sequences=1)
    output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    # If you are getting "translation": "<extra_id_0>", it means the model is outputting a special token instead of a translation.
    # This usually happens if:
    # 1. The model is not fine-tuned for translation (mt5-small is a general model, not specifically trained for English-Swahili).
    # 2. The input prompt format is not what the model expects.
    # 3. The tokenizer/model is not aligned with your fine-tuned checkpoint.

    # Suggestions:
    # - If you have a fine-tuned model, load it instead of "google/mt5-small".
    # - Check your input prompt format. For mT5, try using: "<en> {text} </s>" or "<sw> {text} </s>" depending on your training.
    # - If you only have the base model, it won't translate out-of-the-box.

    # Example fix if you have a custom checkpoint:
    # model = AutoModelForSeq2SeqLM.from_pretrained("model/translator_model.pt")

    # Example prompt adjustment:
    # input_text = f"<en> {request.text} </s>"

    # If you only have the base model, you need a fine-tuned checkpoint for translation.
