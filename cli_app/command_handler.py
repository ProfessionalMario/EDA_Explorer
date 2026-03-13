from rich.table import Table
from rich.console import Console
from data.loader import load_dataset
from data.schema_extractor import extract_schema
from data.registry import DatasetRegistry
from utils.logger import logger
from core.query_router import QueryRouter
from agents.metadata_agent import MetadataAgent
from agents.dataframe_agent import DataFrameAgent
from agents import visualization_agent
router = QueryRouter()
console = Console()
registry = DatasetRegistry()
metadata_agent = MetadataAgent(registry)
dataframe_agent = DataFrameAgent(registry)

def handle_command(command):

    try:

        parts = command.strip().split()

        if not parts:
            return ""

        action = parts[0]
        agent = router.route(command)

        if action == "load":

            if len(parts) < 2:
                return "Please provide a dataset path."

            path = parts[1]

            name, df = load_dataset(path)

            schema = extract_schema(df)

            registry.register_dataset(name, df, schema)

            return f"Dataset '{name}' loaded."

        elif action == "list":

            datasets = registry.list_datasets()

            if not datasets:
                return "No datasets loaded."

            table = Table(title="Datasets")
            table.add_column("Name")

            for d in datasets:
                table.add_row(d)

            console.print(table)

            return ""

        elif action == "info":

            if len(parts) < 2:
                return "Provide dataset name."

            name = parts[1]

            meta = registry.get_info(name)

            # defensive access
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

            table.add_row(
                "Numeric Columns",
                ", ".join(numeric) if numeric else "None"
            )

            table.add_row(
                "Categorical Columns",
                ", ".join(categorical) if categorical else "None"
            )

            table.add_row(
                "Column Types",
                ", ".join([f"{k}:{v}" for k, v in column_types.items()])
            )

            console.print(table)

            return ""

        elif action == "describe":

            if len(parts) < 2:
                return "Provide dataset name."

            name = parts[1]

            df = registry.load_dataframe(name)

            console.print(df.describe().round(2))

            return ""

        elif action == "exit":

            return "exit"
        
        if action == "help":

            table = Table(title="EDA Explorer Commands")

            table.add_column("Command")
            table.add_column("Description")

            table.add_row("load <file_path>", "Load dataset")
            table.add_row("list", "List loaded datasets")

            table.add_row("info <dataset>", "Show dataset metadata")
            table.add_row("describe <dataset>", "Statistical summary")
            table.add_row("columns <dataset>", "Show column names")
            table.add_row("shape <dataset>", "Show dataset size")

            table.add_row("head <dataset> [n]", "Preview first rows")

            table.add_row("NL: show top 10 rows in products", "DataFrameAgent row preview")
            table.add_row("NL: how many rows in customers", "Row count")

            table.add_row("NL: average price in products", "Column mean")
            table.add_row("NL: max price in products", "Column max")
            table.add_row("NL: min price in products", "Column min")

            table.add_row("NL: histogram price in products", "Histogram (numeric column)")
            table.add_row("NL: bar chart category in products", "Bar chart (categorical column)")

            table.add_row("exit", "Quit program")

            console.print(table)

            return
        
        elif action == "columns":
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
        
        elif action == "shape":

            if len(parts) < 2:
                return "Provide dataset name."

            name = parts[1]

            meta = registry.get_info(name)

            rows = meta.get("rows", "unknown")
            cols = len(meta.get("columns", []))

            console.print(f"\nRows: {rows}")
            console.print(f"Columns: {cols}\n")

            return ""
        
        elif action == "head":

            if len(parts) < 2:
                return "Provide dataset name."

            name = parts[1]

            n = 5

            if len(parts) == 3:
                try:
                    n = int(parts[2])
                except:
                    pass

            df = registry.load_dataframe(name)

            console.print(df.head(n))

            return ""
        
        # AGENT ROUTER LAST
        agent = router.route(command)

        if agent == "metadata_agent":

            result = metadata_agent.handle(command)
            console.print(result)
            return ""

        elif agent == "dataframe_agent":

            result = dataframe_agent.handle(command)
            console.print(result)
            return ""

        elif agent == "visualization_agent":

            result = visualization_agent.handle(command)
            console.print(result)
            return ""

        return "Unknown command"
        
        
    except Exception as e:

        logger.error(f"Command failed: {command} | {e}")
        return f"Error: {e}"