# EDA Explorer

A Python CLI tool for exploratory data analysis (EDA) with agent-based query routing.

## Architecture

### Agents
- **metadata_agent** — Schema & structure queries (columns, types, missing values)
- **dataframe_agent** — Statistics (mean, max, min, row counts, head)
- **visualization_agent** — Charts (histograms, bar charts with cardinality guards)
- **transformer_agent** — Cleaning & transformation (drop duplicates, fill nulls, encode, normalize, rename)

### Core Files
- **main.py** — CLI entry point
- **cli_app/command_handler.py** — Command dispatch and agent routing with column validation
- **core/query_router.py** — Rule-based fallback router (transformer keywords take top priority)
- **core/llm_planner.py** — LLM-powered query planner (Ollama + Gemma 3)
- **data/registry.py** — Dataset persistence & management (original + `_clean` copies)
- **utils/logger.py** — File logging to `logs/eda_explorer.log`

## Query Routing Flow

1. **Schema-aware LLM Planner** (Gemma 3 via Ollama) — Reads dataset JSON metadata (never parquet), injects real column names/types into prompt, returns an exact {agent, operation, dataset, column} plan
2. **Column Validation** — Verifies plan column exists in dataset before dispatch
3. **Plan-based Dispatch** — For transformer_agent, plan is passed directly to skip keyword guessing
4. **Fallback Router** — Rule-based keyword matching when Ollama is unavailable (transformer > metadata > dataframe > visual)

## Routing Priority (fallback router)

Transformer action words are checked first to prevent ambiguity:
- `drop`, `fill`, `normalize`, `encode`, `rename`, `impute`, `strip`, `duplicate` → **transformer_agent**
- `column`, `numeric`, `categorical`, `missing` → **metadata_agent**
- `average`, `mean`, `max`, `min`, `top`, `rows` → **dataframe_agent**
- `hist`, `bar`, `plot`, `chart`, `graph` → **visualization_agent**

## "list" Command Disambiguation

`list` alone → shows all loaded datasets
`list columns/numeric/categorical/fields in <dataset>` → routes to metadata_agent

## Column Validation (LLM plan guard)

After the LLM returns a plan with a column name, the system verifies that column
actually exists in the dataset before dispatching. If not found, returns a clear
error listing available columns.

## Workflow

- **Start application**: `python3 main.py` (interactive CLI)

## Using the LLM Planner with Ollama (Gemma 3)

```bash
# 1. Install Ollama from https://ollama.ai
# 2. Pull Gemma 3 (faster and more reliable than Gemma 2)
ollama pull gemma3

# 3. Run ollama server in a terminal
ollama serve

# 4. Optionally set environment variables in Replit secrets:
#    OLLAMA_MODEL=gemma3   (default)
#    OLLAMA_BASE_URL=http://localhost:11434  (default)

# 5. Run the app
python3 main.py
```

No API key is needed — Ollama runs locally for free.
If Ollama is unavailable, the system automatically falls back to the rule-based router.

## Key Commands & Examples

**Structural queries (metadata_agent):**
- `list columns in leads`
- `list numeric columns in organizations`
- `how many missing values in people`
- `what are the categorical columns in leads`

**Cleaning (transformer_agent):**
- `drop duplicates in leads`
- `fill nulls in organizations`
- `fill industry in leads`
- `drop constant columns in organizations`
- `strip whitespace in people`
- `drop column description in leads`

**Analysis (dataframe_agent):**
- `show top 10 rows in leads`
- `how many rows in organizations`
- `max number of employees in organizations`
- `min founded in organizations`

**Visualization (visualization_agent):**
- `histogram founded in organizations`
- `bar chart industry in leads`

**Transformation (transformer_agent):**
- `normalize founded in organizations`
- `encode industry in organizations`
- `rename industry to sector in organizations`

## Features

✓ **Safe cleaning** — All ops on `<dataset>_clean` copy, originals never modified
✓ **Smart null filling** — Mean/median for numeric (based on skewness), mode for categorical
✓ **Visualization guardrails** — Blocks bar charts on >50 unique values
✓ **Dual routing** — Gemma 3 LLM planner + rule-based fallback
✓ **Column validation** — LLM plan columns are verified against actual dataset schema
✓ **"list" disambiguation** — Context-aware routing for list commands
✓ **Comprehensive logging** — All operations logged to `logs/eda_explorer.log`

## Test Suites

- **testing/test_agent_routing.py** — 28 routing & validation tests (all agents, list disambiguation, column guard)
- **testing/test_transformer.py** — 19 cleaning & transformation tests
- **testing/test_visualization.py** — 8 visualization tests (charts saved to `test_output/`)

Run routing tests:
```bash
python3 testing/test_agent_routing.py
```

## Demo Outputs

Sample outputs from all agents are stored in `demo/`:
- `metadata_agent_*.txt` — columns, numeric, categorical, missing value queries
- `dataframe_agent_*.txt` — top rows, row counts, max/min stats
- `transformer_agent_*.txt` — drop duplicates, strip whitespace, normalize, encode

## Dependencies

Python 3.12 with: pandas, matplotlib, rich, typer, pyarrow, openpyxl, requests, openai
