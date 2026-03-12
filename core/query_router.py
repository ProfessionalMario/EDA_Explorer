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

    def route(self, query: str):

        q = query.lower()

        try:

            for word in self.metadata_keywords:
                if word in q:
                    return "metadata_agent"

            for word in self.visual_keywords:
                if word in q:
                    return "visualization_agent"

            for word in self.stats_keywords:
                if word in q:
                    return "dataframe_agent"

            return "rag_agent"

        except Exception as e:

            logger.error(f"Routing failed | {e}")

            return "fallback_agent"