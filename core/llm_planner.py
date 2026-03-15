import json
import os
import requests
from utils.logger import logger


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "planner")


class LLMPlanner:
    """
    Query planner using Ollama local LLM.
    """

    def __init__(self):
        self.enabled = True

        logger.info(f"LLMPlanner enabled (Ollama backend, model={OLLAMA_MODEL})")

        self.system_prompt = """
You are a planner for a data analysis CLI system.

Convert the user query into a JSON execution plan.

Return ONLY JSON with exactly this structure:

{
  "agent": "<agent_name>",
  "operation": "<operation>",
  "dataset": "<dataset_name or null>",
  "column": "<column_name or null>"
}

Valid agents:
metadata_agent
dataframe_agent
visualization_agent
transformer_agent

metadata_agent operations:
columns, numeric_columns, categorical_columns, missing_values, column_count

dataframe_agent operations:
head, row_count, mean, max, min

visualization_agent operations:
histogram, bar_chart

transformer_agent operations:
drop_duplicates, fill_nulls, drop_column, drop_constant_columns,
strip_whitespace, normalize, encode, rename

Rules:
- Output ONLY JSON
- No explanation
- No markdown
- If dataset is missing return null
- If column is missing return null
"""

    def _call_ollama(self, user_query):
        """Call Ollama API locally."""
        try:
            prompt = f"{self.system_prompt}\n\nUser Query: {user_query}\n\nJSON:"

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
                        "num_predict": 60,
                        "stop": ["\n\n"]
                    }
                },
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            return result.get("response", "").strip()

        except Exception as e:
            logger.error(f"Ollama call failed | {e}")
            return None

    def plan(self, query):
        """Get agent routing plan from LLM."""

        content = self._call_ollama(query)

        if not content:
            return None

        try:
            # Clean markdown if model returns ```json
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            plan = json.loads(content)

            # ── VALIDATION ─────────────────────────────

            VALID_AGENTS = {
                "metadata_agent",
                "dataframe_agent",
                "visualization_agent",
                "transformer_agent"
            }

            VALID_OPERATIONS = {
                "columns", "numeric_columns", "categorical_columns",
                "missing_values", "column_count",
                "head", "row_count", "mean", "max", "min",
                "histogram", "bar_chart",
                "drop_duplicates", "fill_nulls", "drop_column",
                "drop_constant_columns", "strip_whitespace",
                "normalize", "encode", "rename"
            }

            agent = plan.get("agent")
            operation = plan.get("operation")

            if agent not in VALID_AGENTS:
                logger.error(f"Invalid agent returned: {agent}")
                return None

            if operation not in VALID_OPERATIONS:
                logger.error(f"Invalid operation returned: {operation}")
                return None

            plan.setdefault("dataset", None)
            plan.setdefault("column", None)

            logger.info(f"LLMPlanner → {plan}")
            return plan

        except json.JSONDecodeError as e:
            logger.error(f"LLM response not valid JSON | {e} | content: {content}")
            return None

        except Exception as e:
            logger.error(f"LLMPlanner error | {e}")
            return None