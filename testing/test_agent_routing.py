"""
Agent Routing Test Suite
========================
Tests that every query type routes to the correct agent (rule-based router),
that the 'list' command is correctly disambiguated, and that the LLM plan
column-validation guard works.

No Ollama required — all tests use the rule-based fallback router directly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch
from core.query_router import QueryRouter
from data.registry import DatasetRegistry
from cli_app.command_handler import _validate_plan_column, _is_list_with_context

router = QueryRouter()

passed = 0
failed = 0


def run_test(label, got, expected):
    global passed, failed
    ok = got == expected
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {label}")
    print(f"       Expected : {expected}")
    print(f"       Got      : {got}\n")
    if ok:
        passed += 1
    else:
        failed += 1


def run_bool_test(label, got, expected=True):
    global passed, failed
    ok = bool(got) == expected
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {label}")
    print(f"       Expected : {expected}")
    print(f"       Got      : {got}\n")
    if ok:
        passed += 1
    else:
        failed += 1


print("=" * 60)
print("  Agent Routing Test Suite")
print("=" * 60)


# ── METADATA AGENT ────────────────────────────────────────────
print("\n--- Metadata Agent Routing ---\n")

run_test("Columns query",
         router.route("show all columns in leads"), "metadata_agent")
run_test("Numeric columns query",
         router.route("what are the numeric columns in leads"), "metadata_agent")
run_test("Categorical columns query",
         router.route("list categorical columns in organizations"), "metadata_agent")
run_test("Missing values query",
         router.route("how many missing values in people"), "metadata_agent")
run_test("Schema query",
         router.route("show schema for organizations"), "metadata_agent")


# ── DATAFRAME AGENT ───────────────────────────────────────────
print("--- DataFrame Agent Routing ---\n")

run_test("Average query",
         router.route("average annual_revenue in leads"), "dataframe_agent")
run_test("Mean query",
         router.route("mean of employees in organizations"), "dataframe_agent")
run_test("Max query",
         router.route("max annual_revenue in leads"), "dataframe_agent")
run_test("Min query",
         router.route("min employees in organizations"), "dataframe_agent")
run_test("Top rows query",
         router.route("show top 10 rows in leads"), "dataframe_agent")
run_test("Row count query",
         router.route("how many rows in leads"), "dataframe_agent")


# ── VISUALIZATION AGENT ───────────────────────────────────────
print("--- Visualization Agent Routing ---\n")

run_test("Histogram query",
         router.route("histogram of annual_revenue in leads"), "visualization_agent")
run_test("Bar chart query",
         router.route("bar chart of industry in leads"), "visualization_agent")
run_test("Plot query",
         router.route("plot distribution in organizations"), "visualization_agent")
run_test("Graph query",
         router.route("graph of employees"), "visualization_agent")


# ── TRANSFORMER AGENT — existing ops ─────────────────────────
print("--- Transformer Agent Routing (existing ops) ---\n")

run_test("Drop duplicates",
         router.route("drop duplicates in leads"), "transformer_agent")
run_test("Fill nulls",
         router.route("fill nulls in organizations"), "transformer_agent")
run_test("Normalize",
         router.route("normalize annual_revenue in leads"), "transformer_agent")
run_test("Encode",
         router.route("encode industry in leads"), "transformer_agent")
run_test("Rename",
         router.route("rename industry to sector in leads"), "transformer_agent")
run_test("Drop column (no metadata collision)",
         router.route("drop column description in leads"), "transformer_agent")
run_test("Impute (no metadata collision)",
         router.route("impute missing in organizations"), "transformer_agent")
run_test("Strip whitespace",
         router.route("strip whitespace in people"), "transformer_agent")


# ── TRANSFORMER AGENT — new preprocessing ops ─────────────────
print("--- Transformer Agent Routing (new preprocessing ops) ---\n")

run_test("Standardize",
         router.route("standardize number of employees in organizations"), "transformer_agent")
run_test("Z-score keyword",
         router.route("z-score normalize founded in organizations"), "transformer_agent")
run_test("Zscore keyword",
         router.route("zscore the index column in leads"), "transformer_agent")
run_test("One-hot encoding",
         router.route("one hot encode industry in organizations"), "transformer_agent")
run_test("Onehot keyword",
         router.route("onehot encode sex in people"), "transformer_agent")
run_test("Dummies keyword",
         router.route("get dummies for industry in organizations"), "transformer_agent")
run_test("Fill with mean",
         router.route("fill with mean in organizations"), "transformer_agent")
run_test("Fill with median",
         router.route("fill nulls with median in leads"), "transformer_agent")
run_test("Fill with mode",
         router.route("fill missing using mode in people"), "transformer_agent")
run_test("Fill zero",
         router.route("fill with zero in leads"), "transformer_agent")
run_test("Drop missing rows",
         router.route("drop missing rows in organizations"), "transformer_agent")
run_test("Drop missing cols",
         router.route("drop missing columns in leads"), "transformer_agent")
run_test("Dropna keyword",
         router.route("dropna in organizations"), "transformer_agent")


# ── LIST DISAMBIGUATION ───────────────────────────────────────
print("--- List Ambiguity Detection ---\n")

run_bool_test("'list columns in leads' → metadata context",
              _is_list_with_context("list columns in leads"), expected=True)
run_bool_test("'list all numeric columns in people' → metadata context",
              _is_list_with_context("list all numeric columns in people"), expected=True)
run_bool_test("'list' alone → no context (dataset list)",
              _is_list_with_context("list"), expected=False)
run_bool_test("'list datasets' → no context (dataset list)",
              _is_list_with_context("list datasets"), expected=False)


# ── COLUMN VALIDATION ─────────────────────────────────────────
print("--- Column Validation (LLM plan guard) ---\n")

registry = DatasetRegistry()
datasets = registry.list_datasets()

if datasets:
    sample_dataset = [d for d in datasets if not d.endswith("_clean")][0]
    info = registry.get_info(sample_dataset)
    real_columns = info.get("columns", [])

    if real_columns:
        real_col = real_columns[0]

        with patch("cli_app.command_handler.registry", registry):
            ok, _ = _validate_plan_column({
                "agent": "transformer_agent", "operation": "fill_mean",
                "dataset": sample_dataset, "column": real_col
            })
            run_bool_test(f"Valid column '{real_col}' in '{sample_dataset}' → passes",
                          ok, expected=True)

            ok, _ = _validate_plan_column({
                "agent": "transformer_agent", "operation": "standardize",
                "dataset": sample_dataset, "column": "ghost_col_xyz"
            })
            run_bool_test("Non-existent column 'ghost_col_xyz' → fails validation",
                          not ok, expected=True)

            ok, _ = _validate_plan_column({
                "agent": "transformer_agent", "operation": "drop_missing_rows",
                "dataset": sample_dataset, "column": None
            })
            run_bool_test("Plan with column=None → always passes",
                          ok, expected=True)
else:
    print("[SKIP] No datasets loaded — skipping column validation tests\n")


# ── SUMMARY ───────────────────────────────────────────────────
print("=" * 60)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed.")
print("=" * 60)
