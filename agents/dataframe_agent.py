from utils.logger import logger


class DataFrameAgent:

    def __init__(self, registry):
        self.registry = registry


    def _detect_dataset(self, query, datasets):
        """
        Detect dataset name from query.
        Falls back to first dataset if none mentioned.
        """
        q = query.lower()

        for d in datasets:
            if d.lower() in q:
                return d

        logger.info("Dataset not specified, using default dataset.")
        return datasets[0]


    def _detect_column(self, query, columns):
        """
        Detect column name from query.
        """
        q = query.lower()

        for col in columns:
            if col.lower() in q:
                return col

        return None


    def _detect_number(self, query, default=5):
        """
        Extract number from query (used for top N rows).
        """
        words = query.split()

        for w in words:
            if w.isdigit():
                return int(w)

        return default


    def handle(self, query):

        q = query.lower()

        try:

            datasets = self.registry.list_datasets()

            if not datasets:
                logger.warning("DataFrameAgent called with no datasets loaded.")
                return "No datasets available."

            dataset = self._detect_dataset(q, datasets)

            df = self.registry.load_dataframe(dataset)

            columns = df.columns.tolist()

        except Exception as e:
            logger.error(f"Failed loading dataset in DataFrameAgent | {e}")
            return "Failed to load dataset."


        try:

            # -------- SHOW ROWS --------
            if "top" in q or "first" in q:

                n = self._detect_number(q, default=5)

                logger.info(f"Showing first {n} rows from {dataset}")

                return df.head(n)


            # -------- ROW COUNT --------
            if "how many rows" in q or "row count" in q or "count rows" in q:

                logger.info(f"Row count requested for {dataset}")

                return f"{dataset} has {len(df)} rows."


            # -------- COLUMN DETECTION --------
            column = self._detect_column(q, columns)

            if column is None and any(
                word in q for word in ["average", "mean", "max", "min", "highest", "lowest"]
            ):
                logger.warning("Column not detected for dataframe operation.")
                return "Column not found in dataset."


            # -------- MEAN / AVERAGE --------
            if "average" in q or "mean" in q:

                result = df[column].mean()

                logger.info(f"Mean computed for {column} in {dataset}")

                return f"Average {column} in {dataset}: {round(result, 2)}"


            # -------- MAX --------
            if "max" in q or "highest" in q:

                result = df[column].max()

                logger.info(f"Max computed for {column} in {dataset}")

                return f"Max {column} in {dataset}: {result}"


            # -------- MIN --------
            if "min" in q or "lowest" in q:

                result = df[column].min()

                logger.info(f"Min computed for {column} in {dataset}")

                return f"Min {column} in {dataset}: {result}"


            return "DataFrame query not understood."


        except Exception as e:

            logger.error(f"DataFrame operation failed | Query: {query} | Error: {e}")

            return "DataFrame agent error."