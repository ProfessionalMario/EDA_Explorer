from utils.logger import logger


class QueryRouter:

    def __init__(self):

        self.metadata_keywords = [
            "columns",
            "shape",
            "info",
            "describe"
        ]

        self.visual_keywords = [
            "plot",
            "graph",
            "scatter",
            "hist"
        ]

        self.stats_keywords = [
            "average",
            "mean",
            "median",
            "sum",
            "count"
        ]

    def route(self, query):

            q = query.lower()

            # Metadata queries
            metadata_keywords = [
                "column",
                "numeric",
                "categorical",
                "missing"
            ]

            # Dataframe analysis queries
            dataframe_keywords = [
                "average",
                "mean",
                "max",
                "min",
                "top",
                "count",
                "rows"
            ]

            if any(word in q for word in metadata_keywords):
                logger.info("Routing → metadata_agent")
                return "metadata_agent"

            if any(word in q for word in dataframe_keywords):
                logger.info("Routing → dataframe_agent")
                return "dataframe_agent"
            
            if any(word in q for word in ["hist", "histogram", "bar", "plot"]):
                return "visualization_agent"
            
            return "unknown command"