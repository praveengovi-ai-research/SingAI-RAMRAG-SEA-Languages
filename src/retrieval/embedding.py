import numpy as np
from tqdm.auto import tqdm
from openai import OpenAI
from src.config.settings import EMBED_MODEL

def openai_embed_norm(texts, api_key, batch_size=128):
    client = OpenAI(api_key=api_key)
    vecs = []
    # Ensure texts are strings
    texts = [str(x) for x in texts]
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            X = np.array([d.embedding for d in resp.data], dtype=np.float32)
            # Normalize
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            vecs.append(X)
        except Exception as e:
            print(f"Error embedding batch {i}: {e}")
            raise e
            
    if not vecs:
        return np.array([], dtype=np.float32)
        
    return np.vstack(vecs)
