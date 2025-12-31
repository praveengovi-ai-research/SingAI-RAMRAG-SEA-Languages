import pandas as pd
from datasets import load_dataset
from src.config.settings import HF_KB_REPO, HF_EVAL_REPO

def load_kb_data():
    """Loads the Knowledge Base dataset."""
    print(f"Loading KB from {HF_KB_REPO}...")
    kb_df = load_dataset(HF_KB_REPO, split="train").to_pandas()
    
    # Preprocess text
    kb_df["kb_text"] = (
        kb_df["title"].fillna("").astype(str).str.strip()
        + "\n\n"
        + kb_df["content"].fillna("").astype(str).str.strip()
    ).str.strip()
    
    return kb_df

def load_eval_data():
    """Loads the Evaluation dataset."""
    print(f"Loading Eval from {HF_EVAL_REPO}...")
    ev_df = load_dataset(HF_EVAL_REPO, split="train").to_pandas()
    return ev_df
