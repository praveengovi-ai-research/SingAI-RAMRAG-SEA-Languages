import re

# HuggingFace Repos
HF_KB_REPO = "praveengovi/multilingual-smb-sales-care-kb"
HF_EVAL_REPO = "praveengovi/multilingual-smb-sales-care-eval-dataset"

# Model Config
EMBED_MODEL = "text-embedding-3-large"
TOP_K_SEARCH = 5

# Paths
INDEX_PATH = "singai_kb.index"
META_PATH = "phoenix_chunk_meta.parquet"
CHUNK_VEC_DIR = "."
QUERY_VEC_DIR = "."

# Chunking Specifications
CHUNK_SPEC_BY_LANG = {
    "zh":    {"max_chars": 320,  "overlap_chars": 40},
    "ta":    {"max_chars": 700,  "overlap_chars": 80},
    "en":    {"max_chars": 1200, "overlap_chars": 150},
    "en_sg": {"max_chars": 1200, "overlap_chars": 150},
    "ms":    {"max_chars": 1100, "overlap_chars": 140},
    "id":    {"max_chars": 1100, "overlap_chars": 140},
}
DEFAULT_SPEC = {"max_chars": 900, "overlap_chars": 120}

# Sentence Splitters
_SENT_SPLIT = {
    "zh": re.compile(r"(?<=[。！？\n])"),
    "default": re.compile(r"(?<=[.!?\n])"),
}

# Language Scopes
LANG_SCOPE_MAP = {
    "en_sg": ["en_sg", "en"],
    "ms": ["ms", "en"],
    "id": ["id", "en"],
    "ta": ["ta", "en"],
    "zh": ["zh", "en"],
}
