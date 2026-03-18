import json
from pathlib import Path
import pandas as pd
from utils.logger import logger
import os 
DATASET_DIR = Path("data/datasets")
METADATA_DIR = Path("data/metadata")

DATASET_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


class DatasetRegistry:

    def __init__(self):

        self.datasets = {}

        self._load_existing()

    def _load_existing(self):

        try:

            for meta_file in METADATA_DIR.glob("*.json"):

                name = meta_file.stem

                with open(meta_file, "r") as f:
                    metadata = json.load(f)

                self.datasets[name] = metadata
                logger.info(f"Loaded {len(self.datasets)} datasets into registry")

        except Exception as e:
            logger.error(f"Registry loading failed | {e}")

    def delete_dataset(self, name):
        try:
            deleted_files = []

            dataset_dir = "data/datasets"
            metadata_dir = "data/metadata"

            # Possible variations
            variants = [name, f"{name}_clean"]

            for variant in variants:
                parquet_path = os.path.join(dataset_dir, f"{variant}.parquet")
                metadata_path = os.path.join(metadata_dir, f"{variant}.json")

                if os.path.exists(parquet_path):
                    os.remove(parquet_path)
                    deleted_files.append(parquet_path)

                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    deleted_files.append(metadata_path)

            if not deleted_files:
                return f"No dataset found for '{name}'"

            logger.info(f"Deleted dataset {name} | Files: {deleted_files}")

            return f"Deleted dataset '{name}' successfully."

        except Exception as e:
            logger.error(f"Delete failed | {e}")
            return f"Failed to delete dataset '{name}'"

    def register_dataset(self, name, df, schema):

        try:

            if name in self.datasets:
                raise ValueError(f"Dataset '{name}' already exists")

            parquet_path = DATASET_DIR / f"{name}.parquet"
            meta_path = METADATA_DIR / f"{name}.json"

            df.to_parquet(parquet_path)

            with open(meta_path, "w") as f:
                json.dump(schema, f, indent=2)

            self.datasets[name] = schema

            logger.info(f"Dataset registered | {name}")

        except Exception as e:
            logger.error(f"Dataset registration failed | {e}")
            raise

    def dataset_exists(self, name):

        return name in self.datasets

    def list_datasets(self):

        return list(self.datasets.keys())

    def get_info(self, name):

        if name not in self.datasets:
            raise ValueError("Dataset not found")

        return self.datasets[name]

    def update_dataset(self, name, df, schema):

        try:

            parquet_path = DATASET_DIR / f"{name}.parquet"
            meta_path = METADATA_DIR / f"{name}.json"

            df.to_parquet(parquet_path)

            with open(meta_path, "w") as f:
                json.dump(schema, f, indent=2)

            self.datasets[name] = schema

            logger.info(f"Dataset updated | {name}")

        except Exception as e:
            logger.error(f"Dataset update failed | {e}")
            raise

    def load_dataframe(self, name, sample=True, sample_size=50000):
        try:
            # ---------- VALIDATION ----------
            if name not in self.datasets:
                logger.error(f"Dataset '{name}' not found in registry")
                raise ValueError(f"Dataset '{name}' not found")

            path = DATASET_DIR / f"{name}.parquet"

            if not path.exists():
                logger.error(f"Parquet file missing: {path}")
                raise FileNotFoundError(f"{path} not found")

            logger.info(f"Loading dataset: {name}")

            # ---------- LOAD ----------
            df = pd.read_parquet(path)

            logger.info(f"Loaded dataset '{name}' | shape={df.shape}")

            # ---------- SMART SAMPLING ----------
            if sample and len(df) > sample_size:
                logger.info(
                    f"Dataset '{name}' is large ({len(df)} rows). "
                    f"Sampling {sample_size} rows for analysis."
                )

                df = df.sample(sample_size, random_state=42)

                logger.info(f"Sampled dataset '{name}' | new_shape={df.shape}")

            return df

        except Exception as e:
            logger.error(f"Failed to load dataset '{name}' | {e}")
            raise