import json
from pathlib import Path
import pandas as pd
from utils.logger import logger

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

    def load_dataframe(self, name):

        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")

        path = DATASET_DIR / f"{name}.parquet"

        return pd.read_parquet(path)