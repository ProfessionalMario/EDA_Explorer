from utils.logger import logger
import matplotlib.pyplot as plt


class VisualizationAgent:

    def __init__(self, registry):
        self.registry = registry


    def _detect_dataset(self, query, datasets):

        q = query.lower()

        for d in datasets:
            if d.lower() in q:
                return d

        logger.info("Dataset not specified, using default dataset.")
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
                logger.warning("VisualizationAgent called with no datasets loaded.")
                return "No datasets available."

            dataset = self._detect_dataset(q, datasets)

            df = self.registry.load_dataframe(dataset)

            columns = df.columns.tolist()

        except Exception as e:

            logger.error(f"Failed loading dataset in VisualizationAgent | {e}")

            return "Failed to load dataset."

        try:

            column = self._detect_column(q, columns)

            if column is None:
                logger.warning("Column not detected for visualization.")
                return "Column not found in dataset."

            # ---------- HISTOGRAM ----------
            if "hist" in q or "histogram" in q:

                logger.info(f"Generating histogram for {column} in {dataset}")

                plt.figure()
                df[column].hist()
                plt.title(f"Histogram of {column}")
                plt.xlabel(column)
                plt.ylabel("Frequency")
                plt.show()

                return "Histogram generated."


            # ---------- BAR CHART ----------
            if "bar" in q or "bar chart" in q:

                logger.info(f"Generating bar chart for {column} in {dataset}")

                plt.figure()
                df[column].value_counts().plot(kind="bar")
                plt.title(f"Bar Chart of {column}")
                plt.xlabel(column)
                plt.ylabel("Count")
                plt.show()

                return "Bar chart generated."


            return "Visualization query not understood."


        except Exception as e:

            logger.error(f"Visualization failed | Query: {query} | Error: {e}")

            return "Visualization agent error."