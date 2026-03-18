from rich.table import Table
from rich.console import Console
from data.loader import load_dataset
from data.schema_extractor import extract_schema
from data.registry import DatasetRegistry
from utils.logger import logger
from core.query_router import QueryRouter
from agents.metadata_agent import MetadataAgent
from agents.dataframe_agent import DataFrameAgent
from agents.visualization_agent import VisualizationAgent
from agents.transformer_agent import TransformerAgent
from core.llm_planner import LLMPlanner
from agents.analysis_agent import AnalysisAgent
from data.registry import DatasetRegistry
router = QueryRouter()
llm_planner = LLMPlanner()
console = Console()
registry = DatasetRegistry()
metadata_agent = MetadataAgent(registry)
dataframe_agent = DataFrameAgent(registry)
visualization_agent = VisualizationAgent(registry)
transformer_agent = TransformerAgent(registry)
analysis_agent = AnalysisAgent(registry)

METADATA_CONTEXT_WORDS = [
    "column", "columns", "numeric", "categorical", "missing", "fields", "field"
]


def _validate_plan_column(plan):
    """
    If the LLM plan specifies a column, verify it actually exists in the dataset.
    Returns (ok, error_message). ok=True means safe to proceed.
    """
    column = plan.get("column")
    dataset = plan.get("dataset")

    if not column or not dataset:
        return True, None

    try:
        info = registry.get_info(dataset)
        columns = [c.lower() for c in info.get("columns", [])]
        if column.lower() not in columns:
            msg = (
                f"Column '{column}' does not exist in dataset '{dataset}'. "
                f"Available columns: {', '.join(info.get('columns', []))}"
            )
            logger.warning(f"Column validation failed | {msg}")
            return False, msg
    except Exception as e:
        logger.error(f"Column validation error | {e}")
        return False, f"Could not validate column '{column}' in dataset '{dataset}'."

    return True, None


def _is_list_with_context(command):
    """
    Returns True if 'list' is used in a dataset-specific context
    (e.g. 'list all columns in leads') rather than a bare 'list datasets' call.
    """
    q = command.lower()
    return any(word in q for word in METADATA_CONTEXT_WORDS)


def extract_dataset(command, registry):
    datasets = registry.list_datasets()
    words = command.lower().split()

    for word in words:
        for d in datasets:
            if word == d.lower():
                return d
    return None


def handle_command(command):

    try:
        
        parts = command.strip().split()

        if not parts:
            return ""

        action = parts[0].lower()

        # ── LOAD ──────────────────────────────────────────────────────────────
        if action == "load":
            if len(parts) < 2:
                return "Please provide a dataset path."

            path = parts[1]
            name, df = load_dataset(path)
            schema = extract_schema(df)
            registry.register_dataset(name, df, schema)
            return f"Dataset '{name}' loaded."

        # ── LIST ──────────────────────────────────────────────────────────────
        # If the user says "list columns in X" or "list numeric in X" etc.,
        # route to metadata_agent instead of showing all datasets.
        if action == "list":
            if _is_list_with_context(command):
                result = metadata_agent.handle(command)
                console.print(result)
                console.print(registry.list_datasets())
                return ""

            datasets = registry.list_datasets()

            if not datasets:
                return "No datasets loaded."

            table = Table(title="Datasets")
            table.add_column("Name")

            for d in datasets:
                table.add_row(d)

            console.print(table)
            return ""
        
        #── DELETE ──────────────────────────────────────────────────────────────
        if "delete" in command:
            dataset = extract_dataset(command, registry)

            if not dataset:
                return "Please specify dataset to delete (e.g., 'delete leads')"

            return registry.delete_dataset(dataset)
        # ── INFO ──────────────────────────────────────────────────────────────
        if action == "info":
            if len(parts) < 2:
                return "Provide dataset name."

            name = parts[1]
            meta = registry.get_info(name)

            rows = meta.get("rows", "unknown")
            cols = meta.get("columns", [])
            numeric = meta.get("numeric_columns", [])
            categorical = meta.get("categorical_columns", [])
            column_types = meta.get("column_types", {})

            table = Table(title=f"Dataset Info: {name}")
            table.add_column("Property")
            table.add_column("Value")

            table.add_row("Rows", str(rows))
            table.add_row("Columns", str(len(cols)))
            table.add_row("Numeric Columns", ", ".join(numeric) if numeric else "None")
            table.add_row("Categorical Columns", ", ".join(categorical) if categorical else "None")
            table.add_row(
                "Column Types",
                ", ".join([f"{k}:{v}" for k, v in column_types.items()])
            )

            console.print(table)
            return ""

        # ── DESCRIBE ──────────────────────────────────────────────────────────
        if action == "describe":
            if len(parts) < 2:
                return "Provide dataset name."

            name = parts[1]
            df = registry.load_dataframe(name)
            console.print(df.describe().round(2))
            return ""

        # ── EXIT ──────────────────────────────────────────────────────────────
        if action == "exit":
            return "exit"
        
        # ── Analyze ──────────────────────────────────────────────────────────────
        if action in {"analyze", "analyse"}:  
            return analysis_agent.handle(command)
        
        # ── HELP ──────────────────────────────────────────────────────────────
        if action == "help":
            table = Table(title="EDA Explorer Commands")

            table.add_column("Command")
            table.add_column("Description")

            # ---------- DATASET ----------
            table.add_row("load <file_path>", "Load dataset (auto converts to parquet)")
            table.add_row("delete <dataset>", "Delete dataset (parquet + metadata)")
            table.add_row("delete all", "Delete ALL datasets")
            table.add_row("list", "List available datasets")

            # ---------- METADATA ----------
            table.add_row("info <dataset>", "Show dataset metadata")
            table.add_row("columns <dataset>", "Show column names")
            table.add_row("shape <dataset>", "Show dataset size")
            table.add_row("list columns in <dataset>", "List columns (metadata agent)")

            # ---------- DATA PREVIEW ----------
            table.add_row("head <dataset> [n]", "Preview first rows")
            table.add_row("describe <dataset>", "Statistical summary")

            # ---------- ANALYSIS ----------
            table.add_row("analyze <dataset>", "Full EDA analysis (quality + warnings)")
            table.add_row("missing <dataset>", "Show missing values")
            table.add_row("duplicates <dataset>", "Show duplicate rows")
            table.add_row("correlation <dataset>", "Correlation matrix")

            # ---------- NATURAL LANGUAGE ----------
            table.add_row("NL: show top 10 rows in <dataset>", "Row preview")
            table.add_row("NL: how many rows in <dataset>", "Row count")
            table.add_row("NL: average <column> in <dataset>", "Column mean")
            table.add_row("NL: histogram <column> in <dataset>", "Histogram")
            table.add_row("NL: bar chart <column> in <dataset>", "Bar chart")

            # ---------- SYSTEM ----------
            table.add_row("exit", "Quit program")

            console.print(table)

        # ── COLUMNS ───────────────────────────────────────────────────────────
        if action == "columns":
            if len(parts) < 2:
                return "Provide dataset name."

            name = parts[1]
            meta = registry.get_info(name)
            cols = meta.get("columns", [])

            table = Table(title=f"Columns: {name}")
            table.add_column("Column Name")

            for col in cols:
                table.add_row(col)

            console.print(table)
            return ""

        # ── SHAPE ─────────────────────────────────────────────────────────────
        if action == "shape":
            if len(parts) < 2:
                return "Provide dataset name."

            name = parts[1]
            meta = registry.get_info(name)
            rows = meta.get("rows", "unknown")
            cols = len(meta.get("columns", []))

            console.print(f"\nRows: {rows}")
            console.print(f"Columns: {cols}\n")
            return ""

        # ── HEAD ──────────────────────────────────────────────────────────────
        if action == "head":
            if len(parts) < 2:
                return "Provide dataset name."

            name = parts[1]
            n = 5

            if len(parts) == 3:
                try:
                    n = int(parts[2])
                except Exception:
                    pass

            df = registry.load_dataframe(name)
            console.print(df.head(n))
            return ""

                # ── AGENT ROUTING ─────────────────────────────────────────────────────
        # LLM planner is tried first; falls back to rule-based router if the
        # key is missing or the LLM call fails.

        plan = llm_planner.plan(command)
        agent_name = plan["agent"] if plan else router.route(command)

        # Column validation: if the LLM suggested a column, confirm it exists
        if plan and plan.get("column"):
            ok, err = _validate_plan_column(plan)
            if not ok:
                return err

        agent_map = {
            "metadata_agent":      metadata_agent,
            "dataframe_agent":     dataframe_agent,
            "visualization_agent": visualization_agent,
            "transformer_agent":   transformer_agent,
            "analysis_agent":      analysis_agent,
        }

        if agent_name in agent_map:
            agent = agent_map[agent_name]

            # ---- SPECIAL HANDLING ----

            # Transformer agent uses full plan
            if agent_name == "transformer_agent" and plan:
                result = agent.handle(command, plan=plan)

            #  Analysis agent gets dataset directly
            elif agent_name == "analysis_agent":
                dataset = plan.get("dataset") if plan else None

                # fallback if dataset missing
                if not dataset:
                    datasets = registry.list_datasets()
                    if not datasets:
                        return "No datasets available."
                    dataset = datasets[0]

                result = agent.handle(dataset)

            # Default agents
            else:
                result = agent.handle(command)

            console.print(result)
            return ""

        return "Unknown command. Type 'help' to see available commands."
    except Exception as e:
        logger.error(f"Command failed: {command} | {e}")
        return f"Error: {e}"
