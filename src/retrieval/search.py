import pandas as pd
import numpy as np
import faiss
import os
from src.config.settings import LANG_SCOPE_MAP, META_PATH, CHUNK_VEC_DIR
from src.retrieval.embedding import openai_embed_norm

class RiskAwareRetriever:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chunk_meta_df = None
        self.chunk_vecs = None
        self.lang_indices = {}
        self.lang_scopes = {}
        
        self.load_resources()

    def load_resources(self):
        # Load Metadata
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"Metadata file {META_PATH} not found. Run ingestion first.")
        
        print("Loading metadata...")
        self.chunk_meta_df = pd.read_parquet(META_PATH)
        
        # Load Vectors (In a real scenario, these would be loaded from disk or re-computed)
        # For this setup, we assume they need to be computed or loaded.
        # The notebook computed them in memory.
        # To make this persistent, we should check if they exist.
        
        vec_path = os.path.join(CHUNK_VEC_DIR, "chunk_vecs.npy")
        if os.path.exists(vec_path):
             print(f"Loading vectors from {vec_path}...")
             self.chunk_vecs = np.load(vec_path)
        else:
             print("Vectors not found on disk. In a production system, these should be persisted.")
             print("Re-computing vectors for all chunks (this may take time/cost)...")
             # NOTE: This is dangerous if the index is huge.
             # Ideally we load from keys.
             # For the purpose of this repo structure, we'll provide a placeholder or re-compute.
             # We'll re-compute using the helper.
             texts = self.chunk_meta_df["chunk_text"].tolist()
             self.chunk_vecs = openai_embed_norm(texts, self.api_key)
             np.save(vec_path, self.chunk_vecs)

        self.build_indices()

    def build_indices(self):
        print("Building language-scoped indices...")
        dim = self.chunk_vecs.shape[1]
        chunk_langs = self.chunk_meta_df["lang"].fillna("").astype(str).to_numpy()
        
        # We build indices for known languages in the scope map, plus generally
        unique_langs = set(LANG_SCOPE_MAP.keys())
        unique_langs.update(chunk_langs)
        
        for q_lang in unique_langs:
            if not q_lang: continue
            
            # Get valid target languages for this query language
            scope_langs = LANG_SCOPE_MAP.get(q_lang, [q_lang])
            
            # Filter chunks
            global_idxs = np.where(np.isin(chunk_langs, scope_langs))[0].astype(np.int64)
            
            if len(global_idxs) == 0:
                continue
                
            idx = faiss.IndexFlatIP(dim)
            idx.add(self.chunk_vecs[global_idxs])
            
            self.lang_indices[q_lang] = idx
            self.lang_scopes[q_lang] = global_idxs
            
    def retrieve(self, query: str, lang: str = "en", top_k: int = 5):
        # Embed query
        q_vec = openai_embed_norm([query], self.api_key)[0].reshape(1, -1)
        
        # Select index
        idx = self.lang_indices.get(lang)
        scope_idxs = self.lang_scopes.get(lang)
        
        if idx is None:
            # Fallback to English or all?
            # For now, simplistic fallback to first available or error
            if "en" in self.lang_indices:
                idx = self.lang_indices["en"]
                scope_idxs = self.lang_scopes["en"]
            else:
                return [], []

        # Search
        scores, local_idxs = idx.search(q_vec, top_k)
        
        # Map back to global IDs
        # local_idxs is [1, k]
        global_ids = scope_idxs[local_idxs[0]]
        
        results = []
        for g_id, score in zip(global_ids, scores[0]):
            meta = self.chunk_meta_df.iloc[g_id]
            results.append({
                "score": float(score),
                "text": meta["chunk_text"],
                "doc_id": meta["doc_id"],
                "base_id": meta["base_id"]
            })
            
        return results
