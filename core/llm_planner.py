import json
import os
import requests
from pathlib import Path
from utils.logger import logger


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "hf.co/bartowski/gemma-2-2b-it-GGUF:Q5_K_M")


class LLMPlanner:
    """
    Schema-aware query planner using Ollama (default: gemma3).

    Before calling the LLM, the planner reads the dataset's JSON metadata
    (fast — no parquet touch) and injects column names and types into the
    prompt so the model can resolve natural-language column references to
    their exact names and pick the right preprocessing operation.

    Falls back gracefully if Ollama is unavailable.
    """

    VALID_AGENTS = {
        "metadata_agent",
        "dataframe_agent",
        "visualization_agent",
        "transformer_agent",
        "analysis_agent",
    }

    VALID_OPERATIONS = {
        # metadata
        "columns", "numeric_columns", "categorical_columns",
        "missing_values", "column_count",
        # dataframe
        "head", "row_count", "mean", "max", "min",
        # visualization
        "histogram", "bar_chart",
        # transformer — cleaning
        "drop_duplicates", "fill_nulls", "drop_column",
        "drop_constant_columns", "strip_whitespace",
        "drop_missing_rows", "drop_missing_cols",
        # transformer — explicit fill strategies
        "fill_mean", "fill_median", "fill_mode", "fill_zero",
        # transformer — transforms
        "normalize", "standardize", "encode", "onehot", "rename",
        # analysis
        "analyze","analyse"
    }

    SYSTEM_PROMPT = """\
You are a planner for a data analysis CLI system.

Convert the user query into a JSON execution plan.

Return ONLY valid JSON with exactly this structure:
{
  "agent": "<agent_name>",
  "operation": "<operation>",
  "dataset": "<exact dataset name or null>",
  "column": "<exact column name from schema or null>"
}

Valid agents:
  metadata_agent      — schema / structure queries
  dataframe_agent     — statistics, row previews
  visualization_agent — charts
  transformer_agent   — cleaning, filling, encoding, scaling

metadata_agent operations:
  columns, numeric_columns, categorical_columns, missing_values, column_count

dataframe_agent operations:
  head, row_count, mean, max, min

visualization_agent operations:
  histogram, bar_chart

transformer_agent operations:
  Cleaning  : drop_duplicates, drop_column, drop_constant_columns,
              strip_whitespace, drop_missing_rows, drop_missing_cols
  Filling   : fill_nulls (smart — auto picks mean/median/mode),
              fill_mean, fill_median, fill_mode, fill_zero
  Scaling   : normalize (min-max [0,1]), standardize (z-score)
  Encoding  : encode (label), onehot (one-hot / get_dummies)
  Other     : rename

Rules:
- Output ONLY JSON — no explanation, no markdown, no extra keys.
- Use the EXACT column name from the schema context provided.
- If the query covers all columns (e.g. "fill all nulls"), set column to null.
- For queries about listing/showing structure → metadata_agent.
- For queries about previewing data or computing statistics → dataframe_agent.
- For fill operations: choose fill_mean/fill_median/fill_mode/fill_zero when the
  user explicitly names a strategy; use fill_nulls when they don't.
"""

    def __init__(self):
        self.enabled = True
        logger.info(f"LLMPlanner ready | model={OLLAMA_MODEL} | base={OLLAMA_BASE_URL}")

    # ── schema context ─────────────────────────────────────────────────────

    def _load_schema_context(self, query):
        """
        Scan the query for a known dataset name and load its JSON metadata.
        Returns a compact schema string for injection into the LLM prompt.
        Reads only the tiny JSON file — parquet is never touched.
        """
        meta_dir = Path("data/metadata")
        if not meta_dir.exists():
            return None, ""

        q = query.lower()
        for meta_file in sorted(meta_dir.glob("*.json")):
            name = meta_file.stem
            if name.endswith("_clean"):
                continue
            if name.lower() not in q:
                continue
            try:
                with open(meta_file) as f:
                    schema = json.load(f)

                cols      = schema.get("columns", [])
                numeric   = schema.get("numeric_columns", [])
                cats      = schema.get("categorical_columns", [])
                col_types = schema.get("column_types", {})
                rows      = schema.get("rows", "?")

                lines = [
                    f"Dataset '{name}' ({rows} rows, {len(cols)} columns):",
                    f"  All columns  : {', '.join(cols)}",
                    f"  Numeric      : {', '.join(numeric) if numeric else 'none'}",
                    f"  Categorical  : {', '.join(cats) if cats else 'none'}",
                    f"  Column types : {', '.join(f'{k}:{v}' for k, v in col_types.items())}",
                ]
                return name, "\n".join(lines)
            except Exception as e:
                logger.warning(f"Schema load failed for '{name}' | {e}")

        return None, ""

    # ── ollama call ────────────────────────────────────────────────────────

    def _call_ollama(self, user_query, schema_context=""):
        """POST to local Ollama API and return the raw response string."""
        try:
            schema_block = (
                f"\n\nSchema context (use exact column names from here):\n{schema_context}"
                if schema_context else ""
            )
            prompt = f"{self.SYSTEM_PROMPT}{schema_block}\n\nUser Query: {user_query}\n\nJSON:"

            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "top_p": 0.9,
                        "num_predict": 100,
                        "stop": ["\n\n"],
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()

        except Exception as e:
            logger.error(f"Ollama call failed | {e}")
            return None

    # ── public API ─────────────────────────────────────────────────────────

    def plan(self, query):
        """
        Return a validated execution plan dict, or None if unavailable.
        The plan always contains: agent, operation, dataset, column.
        """
        _, schema_ctx = self._load_schema_context(query)
        content = self._call_ollama(query, schema_ctx)

        if not content:
            return None

        try:
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            plan = json.loads(content)

            agent     = plan.get("agent")
            operation = plan.get("operation")

            if agent not in self.VALID_AGENTS:
                logger.error(f"LLM returned invalid agent: {agent!r}")
                return None

            if operation not in self.VALID_OPERATIONS:
                logger.error(f"LLM returned invalid operation: {operation!r}")
                return None

            plan.setdefault("dataset", None)
            plan.setdefault("column", None)

            logger.info(f"LLMPlanner plan → {plan}")
            return plan

        except json.JSONDecodeError as e:
            logger.error(f"LLM response not valid JSON | {e} | raw: {content!r}")
            return None
        except Exception as e:
            logger.error(f"LLMPlanner error | {e}")
            return None
