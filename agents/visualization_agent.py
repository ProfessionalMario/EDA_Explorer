from utils.logger import logger
import matplotlib.pyplot as plt
import plotext as plt_terminal
import os


class VisualizationAgent:

    def __init__(self, registry):
        self.registry = registry
        os.makedirs("output", exist_ok=True)


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

                values = df[column].dropna().values

                # Terminal plot
                plt_terminal.clear_figure()
                plt_terminal.hist(values, bins=20)
                plt_terminal.title(f"Histogram of {column}")
                plt_terminal.xlabel(column)
                plt_terminal.ylabel("Frequency")
                plt_terminal.show()

                # Save PNG
                filepath = f"output/{dataset}_{column}_hist.png"

                plt.figure()
                df[column].dropna().hist()
                plt.title(f"Histogram of {column}")
                plt.xlabel(column)
                plt.ylabel("Frequency")
                plt.savefig(filepath)
                plt.close()

                logger.info(f"Histogram saved → {filepath}")

                return f"Histogram generated in terminal. PNG saved to {filepath}"


            # ---------- BAR CHART ----------
            if "bar" in q or "bar chart" in q:

                unique_values = df[column].nunique()

                if unique_values > 50:
                    logger.warning(
                        f"Column '{column}' has {unique_values} unique values. Skipping bar chart."
                    )
                    return f"Column '{column}' has {unique_values} unique values. Too many to visualize meaningfully."

                logger.info(f"Generating bar chart for {column} in {dataset}")

                counts = df[column].value_counts()

                # Terminal plot
                plt_terminal.clear_figure()
                plt_terminal.bar(
                    counts.index.astype(str).tolist(),
                    counts.values.tolist()
                )
                plt_terminal.title(f"Bar Chart of {column}")
                plt_terminal.xlabel(column)
                plt_terminal.ylabel("Count")
                plt_terminal.show()

                # Save PNG
                filepath = f"output/{dataset}_{column}_bar.png"

                plt.figure()
                counts.plot(kind="bar")
                plt.title(f"Bar Chart of {column}")
                plt.xlabel(column)
                plt.ylabel("Count")
                plt.savefig(filepath)
                plt.close()

                logger.info(f"Bar chart saved → {filepath}")

                return f"Bar chart generated in terminal. PNG saved to {filepath}"

            return "Visualization query not understood."

        except Exception as e:

            logger.error(f"Visualization failed | Query: {query} | Error: {e}")
            return "Visualization agent error."