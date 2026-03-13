from utils.logger import logger


class DataFrameAgent:

    def __init__(self, registry):

        self.registry = registry

    def _detect_dataset(self, query, datasets):

        q = query.lower()

        for d in datasets:
            if d.lower() in q:
                return d

        # fallback to first dataset
        return datasets[0]  

    def _detect_column(self, query, columns):

        q = query.lower()

        for col in columns:
            if col.lower() in q:
                return col

        return None  
    
    def handle(self, query):

        q = query.lower()

        try:

            datasets = self.registry.list_datasets()

            if not datasets:
                return "No datasets available."

            dataset = self._detect_dataset(q, datasets)

            df = self.registry.load_dataframe(dataset)

            columns = df.columns.tolist()
            if "top" in q or "first" in q:

                n = 5

                words = q.split()

                for w in words:
                    if w.isdigit():
                        n = int(w)
                        break

                return df.head(n)
            
            if "count" in q or "how many rows" in q:

                return f"{dataset} has {len(df)} rows."
            
            if columns:
                column = self._detect_column(q, columns)

            if "average" in q or "mean" in q:

                result = df[column].mean()

                return f"Average {column} in {dataset}: {round(result, 2)}"

            if "max" in q or "highest" in q:

                result = df[column].max()

                return f"Max {column} in {dataset}: {result}"
            
            if "min" in q or "lowest" in q:

                result = df[column].min()

                return f"Min {column} in {dataset}: {result}"
        
            return "Dataframe query not understood."
        

        except Exception as e:

            logger.error(f"DataFrame agent failed | {e}")

            return "DataFrame agent error."