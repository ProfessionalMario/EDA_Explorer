from utils.logger import logger


class QueryRouter:
    """
    Rule-based fallback router.
    Order matters: transformer action words are checked first so that queries
    like 'drop column X' or 'impute missing' don't get swallowed by the
    metadata keyword list.
    """

    # Transformer keywords take top priority — they are explicit action verbs.
    TRANSFORMER_KEYWORDS = [
        "normalize",
        "standardize",
        "zscore",
        "z-score",
        "scale",
        "encode",
        "onehot",
        "one-hot",
        "one hot",
        "dummies",
        "drop",
        "fill",
        "impute",
        "rename",
        "strip",
        "duplicate",
        "constant",
        "whitespace",
        "dropna",
    ]

    # Metadata keywords — structural / schema queries.
    METADATA_KEYWORDS = [
        "column",
        "numeric",
        "categorical",
        "missing",
        "schema",
        "fields",
        "field",
    ]

    # DataFrame / statistics keywords.
    DATAFRAME_KEYWORDS = [
        "average",
        "mean",
        "median",
        "max",
        "min",
        "top",
        "count",
        "rows",
        "sum",
        "highest",
        "lowest",
    ]

    # Visualisation keywords.
    VISUAL_KEYWORDS = [
        "plot",
        "graph",
        "scatter",
        "hist",
        "bar",
        "chart",
        "histogram",
        "distribution",
    ]

    def route(self, query):
        q = query.lower()

        if any(word in q for word in self.TRANSFORMER_KEYWORDS):
            logger.info("Routing → transformer_agent")
            return "transformer_agent"

        if any(word in q for word in self.METADATA_KEYWORDS):
            logger.info("Routing → metadata_agent")
            return "metadata_agent"

        if any(word in q for word in self.DATAFRAME_KEYWORDS):
            logger.info("Routing → dataframe_agent")
            return "dataframe_agent"

        if any(word in q for word in self.VISUAL_KEYWORDS):
            logger.info("Routing → visualization_agent")
            return "visualization_agent"
        
        if "analyze" in query or "analysis" in query or "analyse" in query:
            return "analysis_agent"

        logger.warning(f"No route matched for query: {query}")
        return "unknown command"
