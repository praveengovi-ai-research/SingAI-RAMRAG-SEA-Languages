import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

print("Verifying SingAI-RAMRAG structure...")

try:
    from src.config import settings
    print("✅ src.config.settings imported")
    print(f"   Config check: KB Repo = {settings.HF_KB_REPO}")
except ImportError as e:
    if "No module named" in str(e):
        print(f"⚠️ Missing dependency for config: {e}")
    else:
        print(f"❌ Failed to import src.config.settings: {e}")

try:
    from src.ingestion import loader, indexer
    print("✅ src.ingestion modules imported")
except ImportError as e:
    if "No module named" in str(e):
         print(f"⚠️ Missing dependency for ingestion: {e}")
    else:
         print(f"❌ Failed to import src.ingestion: {e}")

try:
    from src.retrieval import search, embedding
    print("✅ src.retrieval modules imported")
except ImportError as e:
    if "No module named" in str(e):
         print(f"⚠️ Missing dependency for retrieval: {e}")
    else:
         print(f"❌ Failed to import src.retrieval: {e}")

try:
    from src.guardrails.domain_guard import domain_guard
    print("✅ src.guardrails.domain_guard imported")
    # precise check of logic
    if hasattr(domain_guard, 'domain_guard_action'):
         print("   DomainGuard action accessible")
    else:
         print("❌ DomainGuard action NOT found")
except ImportError as e:
    print(f"❌ Failed to import src.guardrails: {e}")

try:
    from src import pipeline
    print("✅ src.pipeline imported")
except ImportError as e:
    if "No module named" in str(e):
         print(f"⚠️ Missing dependency for pipeline: {e}")
    else:
         print(f"❌ Failed to import src.pipeline: {e}")

print("\nFile Structure Check:")
files = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "src/config/settings.py",
    "src/ingestion/indexer.py",
    "src/retrieval/search.py",
    "src/guardrails/domain_guard.py",
    "src/pipeline.py"
]
all_exist = True
for f in files:
    if os.path.exists(f):
        print(f"✅ Found {f}")
    else:
        print(f"❌ Missing {f}")
        all_exist = False

if all_exist:
    print("\nSUCCESS: Project structure matches specification.")
else:
    print("\nFAILURE: Some files are missing.")
