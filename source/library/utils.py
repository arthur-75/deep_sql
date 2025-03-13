import re
import os
from sentence_transformers import SentenceTransformer
import torch

def extract_sql_from_text(text: str) -> str:
    """
    Extracts SQL code enclosed within triple backticks from a given text.

    :param text: The input string containing SQL code.
    :return: Extracted SQL query as a string, or an empty string if no SQL is found.
    """
    pattern = r"```sql\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)  # DOTALL makes '.' match newlines
    
    return match.group(1).strip() if match else ""




"""def load_sentence(model_name: str, hf_tokens: str):


    model = SentenceTransformer(model_name, token=hf_tokens,
                                       trust_remote_code=True)

    return model"""

def load_sentence(name: str, hf_tokens:str ,device=None):
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    # Convert model name into a valid directory name
    model_dir = "models/" + name.replace("/", "_")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Check if model is already saved
    if os.path.exists(model_dir):
        print(f"Loading model from disk: {model_dir}")
        try:
            model = SentenceTransformer(model_dir, device=device, trust_remote_code=True)
            return model
        except Exception as e:
            print(f"Error loading model from {model_dir}, redownloading...\n{e}")

    # Download and save the model
    print(f"Downloading model: {name}")
    model = SentenceTransformer(name, device=device, trust_remote_code=True)
    model.save(model_dir)
    
    return model
