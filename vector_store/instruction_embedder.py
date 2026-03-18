from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

pickle_file = BASE_DIR / "vector_store" / "analyze_embeddings.pkl"
instruction_file = BASE_DIR / "instructions" / "analyze.txt"

def embed_analyze_instructions():
    instruction_file = BASE_DIR / "instructions" / "analyze.txt"
    pickle_file = BASE_DIR / "vector_store" / "analyze_embeddings.pkl"

    # Ensure directory exists
    pickle_file.parent.mkdir(parents=True, exist_ok=True)

    # If embeddings already exist, load
    if pickle_file.exists():
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        # print("Analyze embeddings already exist. Loaded from disk.")
        return data

    # Load instructions
    with open(instruction_file, "r", encoding="utf-8") as f:
        instructions = [line.strip() for line in f if line.strip()]

    # Embed
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(instructions)
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    # Save
    data = {"instructions": instructions, "embeddings": embeddings}
    with open(pickle_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Instruction embeddings created and saved: {len(instructions)} instructions")
    return data


if __name__ == "__main__":
    embed_analyze_instructions()