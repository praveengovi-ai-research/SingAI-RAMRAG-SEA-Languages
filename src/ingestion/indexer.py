import os
import re
import pandas as pd
from typing import List, Dict
from src.config.settings import CHUNK_SPEC_BY_LANG, DEFAULT_SPEC, _SENT_SPLIT, INDEX_PATH, META_PATH
from src.ingestion.loader import load_kb_data
from phoenix_ai.vector_embedding_pipeline import VectorEmbedding
from phoenix_ai.utils import GenAIEmbeddingClient

def _sentences(text: str, lang: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    rx = _SENT_SPLIT.get(lang, _SENT_SPLIT["default"])
    parts = [p.strip() for p in rx.split(text) if p and p.strip()]
    return parts if parts else [text]

def _lang_chunk(text: str, lang: str) -> List[str]:
    spec = CHUNK_SPEC_BY_LANG.get(lang, DEFAULT_SPEC)
    max_chars = spec["max_chars"]
    overlap = spec["overlap_chars"]

    sents = _sentences(text, lang)
    if not sents:
        return []

    chunks_out = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks_out.append(cur)
            cur = s
    if cur:
        chunks_out.append(cur)

    # Apply overlap
    if overlap > 0 and len(chunks_out) > 1:
        overlapped = [chunks_out[0]]
        for i in range(1, len(chunks_out)):
            prev = overlapped[-1]
            tail = prev[-overlap:] if len(prev) > overlap else prev
            overlapped.append((tail + " " + chunks_out[i]).strip())
        chunks_out = overlapped

    return chunks_out

def build_index(api_key: str):
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print(f"Index found at {INDEX_PATH}, skipping build.")
        return

    kb_df = load_kb_data()
    kb_rows = []
    
    print("Chunking documents...")
    for _, r in kb_df.iterrows():
        lang   = str(r.get("lang", "") or "")
        doc_id = str(r.get("doc_id", "") or "")
        base_id= str(r.get("base_id", "") or "")
        region = str(r.get("region", "") or "")
        text   = str(r.get("kb_text", "") or "").strip()

        for j, ch in enumerate(_lang_chunk(text, lang)):
            kb_rows.append({
                "doc_id": doc_id,
                "base_id": base_id,
                "lang": lang,
                "region": region,
                "prechunk_id": j,
                "kb_text_prechunk": ch,
            })

    kb_index_df = pd.DataFrame(kb_rows)
    print(f"Generated {len(kb_index_df)}chunks.")

    # Add metadata prefix for Phoenix
    def _add_meta_prefix(r):
        doc_id  = (r["doc_id"] or "").replace(" ", "_")
        base_id = (r["base_id"] or "").replace(" ", "_")
        lang    = (r["lang"] or "").replace(" ", "_")
        region  = (r["region"] or "").replace(" ", "_")
        return (
            f"__META__ doc_id={doc_id} base_id={base_id} lang={lang} region={region}\n"
            f"{r['kb_text_prechunk']}"
        )

    kb_index_df["kb_text_prechunk"] = kb_index_df.apply(_add_meta_prefix, axis=1)

    print("Building Phoenix Index...")
    embedding_client = GenAIEmbeddingClient(
        provider="openai",
        model="text-embedding-3-large", # Hardcoded in notebook, should use config but keeping consistency
        api_key=api_key
    )
    
    vector = VectorEmbedding(embedding_client, chunk_size=5000, overlap=0)
    
    index_path, chunks = vector.generate_index(
        df=kb_index_df,
        text_column="kb_text_prechunk",
        index_path=INDEX_PATH,
        vector_index_type="local_index"
    )
    
    print(f"Index saved to {index_path}")
    
    # Process metadata again to save as parquet
    # Simplified parsing for now, assuming chunks correspond to kb_index_df roughly
    # In a real run, Phoenix returns 'chunks' as list of texts. 
    # We should re-parse them to get metadata back for the parquet.
    
    META_RE = re.compile(
        r"__META__\s+doc_id=(?P<doc_id>\S+)\s+base_id=(?P<base_id>\S+)\s+lang=(?P<lang>\S+)\s+region=(?P<region>\S+)\s*\n",
        re.I
    )
    
    def parse_chunk_meta(chunk_text: str):
        matches = list(META_RE.finditer(chunk_text or ""))
        if not matches:
             return {"doc_id":"", "base_id":"", "lang":"", "region":"", "chunk_text":(chunk_text or "").strip()}
        
        # Take the most common if multiple matches (though Phoenix usually preserves 1:1 if configured right, but here we prepended)
        # The notebook logic used _mode logic.
        from collections import Counter
        def _mode(xs):
            c = Counter([x for x in xs if x and x.lower() != "nan"])
            return c.most_common(1)[0][0] if c else ""

        doc_ids  = [m.group("doc_id") for m in matches]
        base_ids = [m.group("base_id") for m in matches]
        langs    = [m.group("lang") for m in matches]
        regions  = [m.group("region") for m in matches]
        
        cleaned = META_RE.sub("", chunk_text).strip()
        return {
            "doc_id": _mode(doc_ids),
            "base_id": _mode(base_ids),
            "lang": _mode(langs),
            "region": _mode(regions),
            "chunk_text": cleaned
        }

    chunk_meta_df = pd.DataFrame([parse_chunk_meta(t) for t in chunks])
    chunk_meta_df.to_parquet(META_PATH, index=False)
    print(f"Metadata saved to {META_PATH}")
