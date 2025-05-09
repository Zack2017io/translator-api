import gdown
import os

def download_model():
    url = 'https://drive.google.com/uc?id=1g5TUeFPsVNrOXBfTdpogGYfj8H_5epF5'
    output = 'model/translator_model.pt'
    os.makedirs('model', exist_ok=True)
    gdown.download(url, output, quiet=False)
