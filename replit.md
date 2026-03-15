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
- **cli_app/command_handler.py** — Command dispatch and agent routing
- **core/query_router.py** — Rule-based fallback router (keywords → agent)
- **core/llm_planner.py** — LLM-powered query planner (Ollama or OpenAI)
- **data/registry.py** — Dataset persistence & management (original + `_clean` copies)
- **utils/logger.py** — File logging to `logs/eda_explorer.log`

## Query Routing Flow

1. **LLM Planner** (if enabled) — Uses LLM to understand intent
2. **Fallback Router** — Rule-based keyword matching
3. **Agent Handler** — Executes the appropriate agent

## Workflow

- **Start application**: `python3 main.py` (interactive CLI)

## Using the LLM Planner with Ollama

To enable smart query planning with Ollama:

```bash
# 1. Install Ollama from https://ollama.ai
# 2. Pull a model
ollama pull neural-chat

# 3. Run ollama server in a terminal
ollama serve

# 4. Set environment variables in your Replit secrets:
export LLM_BACKEND=ollama
export OLLAMA_MODEL=neural-chat
export OLLAMA_BASE_URL=http://localhost:11434

# 5. Run the app
python3 main.py
```

The system will now use Ollama for intelligent query planning. If Ollama is unavailable, it automatically falls back to the rule-based router.

## Available Models for Ollama

Popular lightweight models:
- `neural-chat` (7B) — Recommended for general queries
- `mistral` (7B) — Fast, good reasoning
- `llama2` (7B/13B) — Strong general model
- `orca-mini` (3B) — Very small, fast

## Key Commands & Examples

**Cleaning:**
- `drop duplicates in products`
- `fill nulls in products`
- `fill price in products` — Smart fill (mean if symmetric, median if skewed)
- `drop constant columns in products`
- `strip whitespace in products`
- `drop column description in products`

**Analysis:**
- `show top 10 rows in products`
- `average price in products`
- `max price in products`
- `histogram price in products`
- `bar chart category in products`

**Transformation:**
- `normalize price in products` — Min-max to [0, 1]
- `encode category in products` — Label encoding
- `rename stock to inventory in products`

## Features

✓ **Safe cleaning** — All ops on `<dataset>_clean` copy, originals never modified
✓ **Smart null filling** — Mean/median for numeric (based on skewness), mode for categorical
✓ **Visualization guardrails** — Blocks bar charts on >50 unique values
✓ **Dual routing** — LLM planner + rule-based fallback
✓ **Comprehensive logging** — All operations logged to `logs/eda_explorer.log`

## Test Suites

- **test_transformer.py** — 19 cleaning & transformation tests
- **test_visualization.py** — 8 visualization tests (charts saved to `test_output/`)

## Dependencies

Python 3.12 with: pandas, matplotlib, rich, typer, pyarrow, openpyxl, requests (optional for Ollama)
