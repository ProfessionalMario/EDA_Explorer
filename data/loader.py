import pandas as pd
import os
from pathlib import Path

def load_dataset(path):
    #strips path with quotes. 
    path = path.strip().strip('"').strip("'")
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset file not found")
    
    
    name = Path(path).stem

    if path.endswith(".csv"):

        encodings = ["utf-8", "latin1", "cp1252"]

        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc)
                print(f"Loaded CSV using encoding: {enc}")
                return name, df
            except UnicodeDecodeError:
                continue

        raise ValueError("Could not decode CSV file with common encodings.")

    elif path.endswith(".xlsx"):

        df = pd.read_excel(path)
        return name, df

    else:
        raise ValueError("Only CSV and XLSX supported")