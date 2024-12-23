# model_management.py

from transformers import AutoModelForSeq2SeqLM
from config import config

def load_model(model_path=None, device='cuda'):
    if model_path is None:
        model_path = config["model_path"]
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    print(f"Model loaded from {model_path}.")
    return model

def save_model(model, output_dir=None):
    if output_dir is None:
        output_dir = config["finetuned_model_dir"]
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}.")
