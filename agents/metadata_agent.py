from utils.logger import logger


class MetadataAgent:

    def __init__(self, registry):
        self.registry = registry


    def _detect_dataset(self, query, datasets):

        q = query.lower()

        for d in datasets:
            if d.lower() in q:
                return d

        # fallback to first dataset
        return datasets[0]


    def handle(self, query):

        q = query.lower()

        try:

            datasets = self.registry.list_datasets()

            if not datasets:
                return "No datasets available."

            dataset = self._detect_dataset(q, datasets)

            meta = self.registry.get_info(dataset)

            cols = meta.get("columns", [])
            nums = meta.get("numeric_columns", [])
            cats = meta.get("categorical_columns", [])
            miss = meta.get("missing_values", {})

            # ---- INTENT DETECTION ----

            if "how many column" in q or "number of column" in q:
                return f"{dataset} has {len(cols)} columns."

            if "numeric" in q:
                if not nums:
                    return f"No numeric columns found in {dataset}."
                return f"Numeric columns in {dataset}: {', '.join(nums)}"

            if "categorical" in q:
                if not cats:
                    return f"No categorical columns found in {dataset}."
                return f"Categorical columns in {dataset}: {', '.join(cats)}"

            if "missing" in q:
                if not miss:
                    return f"No missing value info for {dataset}."
                return f"Missing values in {dataset}: {miss}"

            if "column" in q:
                if not cols:
                    return f"No columns found for {dataset}."
                return f"Columns in {dataset}: {', '.join(cols)}"
            logger.info(f"MetadataAgent | dataset={dataset} | query={query}")
            return "Metadata query not understood."

        except Exception as e:

            logger.error(f"Metadata agent failed | {e}")

            return "Metadata agent error."