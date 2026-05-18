# from pathlib import Path
# import pickle
# from sentence_transformers import SentenceTransformer

# BASE_DIR = Path(__file__).resolve().parent.parent

# pickle_file = BASE_DIR / "vector_store" / "analyze_embeddings.pkl"
# instruction_file = BASE_DIR / "instructions" / "analyze.txt"

# def embed_analyze_instructions():
#     instruction_file = BASE_DIR / "instructions" / "analyze.txt"
#     pickle_file = BASE_DIR / "vector_store" / "analyze_embeddings.pkl"

#     # Ensure directory exists
#     pickle_file.parent.mkdir(parents=True, exist_ok=True)

#     # If embeddings already exist, load
#     if pickle_file.exists():
#         with open(pickle_file, "rb") as f:
#             data = pickle.load(f)
#         # print("Analyze embeddings already exist. Loaded from disk.")
#         return data

#     # Load instructions
#     with open(instruction_file, "r", encoding="utf-8") as f:
#         instructions = [line.strip() for line in f if line.strip()]

#     # Embed
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings = model.encode(instructions)
#     pickle_file.parent.mkdir(parents=True, exist_ok=True)
#     # Save
#     data = {"instructions": instructions, "embeddings": embeddings}
#     with open(pickle_file, "wb") as f:
#         pickle.dump(data, f)

#     print(f"Instruction embeddings created and saved: {len(instructions)} instructions")
#     return data


# if __name__ == "__main__":
#     embed_analyze_instructions()




import os
import pickle
import requests
from pathlib import Path
from utils.logger import logger

BASE_DIR = Path(__file__).resolve().parent.parent
PICKLE_FILE = BASE_DIR / "vector_store" / "analyze_embeddings.pkl"
INSTRUCTION_FILE = BASE_DIR / "instructions" / "analyze.txt"

def get_ollama_embeddings(texts):
    """Try to get embeddings from local Ollama service."""
    try:
        # Default Ollama address
        url = "http://localhost:11434/api/embed" 
        # Note: Some Ollama versions use /api/embeddings (plural)
        embeddings = []
        for text in texts:
            response = requests.post(
                url, 
                json={"model": "mxbai-embed-large", "input": text}, 
                timeout=5
            )
            embeddings.append(response.json()['embeddings'][0])
        return embeddings
    except Exception:
        return None

def get_hf_api_embeddings(texts):
    """Try to get embeddings via Hugging Face Inference API."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
        
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": texts}, timeout=10)
        return response.json()
    except Exception:
        return None

def embed_analyze_instructions():
    # 1. Ensure directory exists
    PICKLE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 2. Check if cached embeddings exist
    if PICKLE_FILE.exists():
        with open(PICKLE_FILE, "rb") as f:
            return pickle.load(f)

    # 3. Load instructions from file
    if not INSTRUCTION_FILE.exists():
        logger.error(f"Instruction file not found at {INSTRUCTION_FILE}")
        return None

    with open(INSTRUCTION_FILE, "r", encoding="utf-8") as f:
        instructions = [line.strip() for line in f if line.strip()]

    embeddings = None

    # --- FALLBACK LOGIC ---
    
    # Try Ollama First
    logger.info("Attempting Ollama embeddings...")
    embeddings = get_ollama_embeddings(instructions)

    # Try HF API Second
    if embeddings is None:
        logger.info("Ollama failed. Attempting Hugging Face API...")
        embeddings = get_hf_api_embeddings(instructions)

    # Local Heavy Fallback Third
    if embeddings is None:
        logger.warning("External APIs failed. Loading heavy local SentenceTransformer...")
        # Lazy import: Only loads Torch/Transformers if absolutely necessary
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(instructions)

    # 4. Save and Return
    if embeddings is not None:
        data = {"instructions": instructions, "embeddings": embeddings}
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Embeddings saved: {len(instructions)} instructions")
        return data
    
    logger.error("Failed to generate embeddings via any method.")
    return None

if __name__ == "__main__":
    embed_analyze_instructions()