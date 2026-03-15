import shutil
import os
import pandas as pd
from data.registry import DatasetRegistry
from data.schema_extractor import extract_schema
from agents.transformer_agent import TransformerAgent

DATASETS_DIR = "data/datasets"
METADATA_DIR = "data/metadata"
DATASETS_BACKUP = "data/datasets_backup"
METADATA_BACKUP = "data/metadata_backup"

passed = 0
failed = 0


def backup():
    shutil.copytree(DATASETS_DIR, DATASETS_BACKUP, dirs_exist_ok=True)
    shutil.copytree(METADATA_DIR, METADATA_BACKUP, dirs_exist_ok=True)


def restore():
    shutil.rmtree(DATASETS_DIR)
    shutil.rmtree(METADATA_DIR)
    shutil.copytree(DATASETS_BACKUP, DATASETS_DIR)
    shutil.copytree(METADATA_BACKUP, METADATA_DIR)
    shutil.rmtree(DATASETS_BACKUP, ignore_errors=True)
    shutil.rmtree(METADATA_BACKUP, ignore_errors=True)


def fresh():
    return DatasetRegistry(), None


def fresh_agent():
    registry = DatasetRegistry()
    return registry, TransformerAgent(registry)


def run_test(label, query, check_fn, agent):
    global passed, failed
    result = agent.handle(query)
    try:
        ok = check_fn(result, agent)
    except Exception as e:
        ok = False
        print(f"  [check error] {e}")
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {label}")
    print(f"       Query  : {query}")
    print(f"       Result : {result}\n")
    if ok:
        passed += 1
    else:
        failed += 1


print("=" * 60)
print("  Transformer Agent Test Suite")
print("=" * 60)

backup()

try:

    # ── SAFETY: ORIGINAL IS NEVER MODIFIED ─────────────────
    print("--- Safety: original dataset is never modified ---\n")

    registry, agent = fresh_agent()
    original_shape = registry.load_dataframe("products").shape
    agent.handle("drop duplicates in products")
    original_after = registry.load_dataframe("products").shape
    clean_exists = "products_clean" in registry.list_datasets()

    ok = (original_after == original_shape) and clean_exists
    print(f"{'[PASS]' if ok else '[FAIL]'} Original unchanged; products_clean created")
    print(f"       Original shape before : {original_shape}")
    print(f"       Original shape after  : {original_after}")
    print(f"       products_clean exists : {clean_exists}\n")
    passed += ok
    failed += (not ok)

    restore(); backup()

    # ── CLEANING: DROP DUPLICATES ───────────────────────────
    print("--- Cleaning: Drop Duplicates ---\n")

    registry, agent = fresh_agent()
    df = registry.load_dataframe("products")
    df_with_dups = pd.concat([df, df.head(10)], ignore_index=True)
    registry.update_dataset("products", df_with_dups, extract_schema(df_with_dups))

    run_test(
        label="Drop 10 injected duplicate rows",
        query="drop duplicates in products",
        check_fn=lambda result, ag: (
            "dropped 10" in result.lower() and
            ag.registry.load_dataframe("products_clean").duplicated().sum() == 0
        ),
        agent=agent,
    )

    restore(); backup()

    registry, agent = fresh_agent()

    run_test(
        label="No duplicates present → reports 0 dropped",
        query="drop duplicates in products",
        check_fn=lambda result, ag: "dropped 0" in result.lower(),
        agent=agent,
    )

    restore(); backup()

    # ── CLEANING: FILL NULLS ────────────────────────────────
    print("--- Cleaning: Fill Nulls ---\n")

    # symmetric numeric (|skew| < 1) → mean
    registry, agent = fresh_agent()
    df = registry.load_dataframe("products")
    df.loc[0:9, "Price"] = None
    registry.update_dataset("products", df, extract_schema(df))

    run_test(
        label="Fill symmetric Price column → uses mean",
        query="fill price in products",
        check_fn=lambda result, ag: (
            "mean" in result.lower() and
            ag.registry.load_dataframe("products_clean")["Price"].isnull().sum() == 0
        ),
        agent=agent,
    )

    restore(); backup()

    # skewed numeric (|skew| >= 1) → median
    registry, agent = fresh_agent()
    df = registry.load_dataframe("products")
    df["Price"] = df["Price"].astype(float)
    df.loc[0:9, "Price"] = None
    df.loc[10:, "Price"] = df.loc[10:, "Price"] ** 3
    registry.update_dataset("products", df, extract_schema(df))

    run_test(
        label="Fill skewed Price column → uses median",
        query="fill price in products",
        check_fn=lambda result, ag: (
            "median" in result.lower() and
            ag.registry.load_dataframe("products_clean")["Price"].isnull().sum() == 0
        ),
        agent=agent,
    )

    restore(); backup()

    # categorical → mode
    registry, agent = fresh_agent()
    df = registry.load_dataframe("products")
    df.loc[0:9, "Category"] = None
    registry.update_dataset("products", df, extract_schema(df))

    run_test(
        label="Fill categorical Category column → uses mode",
        query="fill category in products",
        check_fn=lambda result, ag: (
            "mode" in result.lower() and
            ag.registry.load_dataframe("products_clean")["Category"].isnull().sum() == 0
        ),
        agent=agent,
    )

    restore(); backup()

    # fill all columns at once
    registry, agent = fresh_agent()
    df = registry.load_dataframe("products")
    df.loc[0:9, "Price"] = None
    df.loc[0:4, "Category"] = None
    registry.update_dataset("products", df, extract_schema(df))

    run_test(
        label="Fill all nulls across every column in one call",
        query="fill nulls in products",
        check_fn=lambda result, ag: (
            "filled" in result.lower() and
            ag.registry.load_dataframe("products_clean").isnull().sum().sum() == 0
        ),
        agent=agent,
    )

    restore(); backup()

    # column with no nulls
    registry, agent = fresh_agent()

    run_test(
        label="Fill column with no nulls → no-op message",
        query="fill price in products",
        check_fn=lambda result, ag: "no missing" in result.lower(),
        agent=agent,
    )

    restore(); backup()

    # ── CLEANING: DROP CONSTANT COLUMNS ────────────────────
    print("--- Cleaning: Drop Constant Columns ---\n")

    # Currency is constant (USD) in the original products data
    registry, agent = fresh_agent()

    run_test(
        label="Drop existing constant column (Currency=USD)",
        query="drop constant columns in products",
        check_fn=lambda result, ag: (
            "currency" in result.lower() and
            "Currency" not in ag.registry.load_dataframe("products_clean").columns
        ),
        agent=agent,
    )

    restore(); backup()

    # inject an additional constant column
    registry, agent = fresh_agent()
    df = registry.load_dataframe("products")
    df["TestConst"] = 0
    registry.update_dataset("products", df, extract_schema(df))

    run_test(
        label="Drop multiple constant columns (Currency + injected TestConst)",
        query="drop constant columns in products",
        check_fn=lambda result, ag: (
            "testconst" in result.lower() and
            "TestConst" not in ag.registry.load_dataframe("products_clean").columns and
            "Currency" not in ag.registry.load_dataframe("products_clean").columns
        ),
        agent=agent,
    )

    restore(); backup()

    # ── CLEANING: STRIP WHITESPACE ──────────────────────────
    print("--- Cleaning: Strip Whitespace ---\n")

    registry, agent = fresh_agent()
    df = registry.load_dataframe("products")
    df["Name"] = "  " + df["Name"].astype(str) + "  "
    registry.update_dataset("products", df, extract_schema(df))

    run_test(
        label="Strip whitespace from string columns",
        query="strip whitespace in products",
        check_fn=lambda result, ag: (
            "stripped" in result.lower() and
            not ag.registry.load_dataframe("products_clean")["Name"]
                .str.startswith(" ").any()
        ),
        agent=agent,
    )

    restore(); backup()

    # ── CLEANING: DROP COLUMN ───────────────────────────────
    print("--- Cleaning: Drop Column ---\n")

    registry, agent = fresh_agent()

    run_test(
        label="Drop Description column",
        query="drop description in products",
        check_fn=lambda result, ag: (
            "dropped" in result.lower() and
            "Description" not in ag.registry.load_dataframe("products_clean").columns
        ),
        agent=agent,
    )

    run_test(
        label="Drop non-existent column → not found",
        query="drop ghostcol in products",
        check_fn=lambda result, ag: "not found" in result.lower(),
        agent=agent,
    )

    restore(); backup()

    # ── TRANSFORMATIONS ─────────────────────────────────────
    print("--- Transformations (secondary) ---\n")

    registry, agent = fresh_agent()

    run_test(
        label="Normalize Price → [0, 1]",
        query="normalize price in products",
        check_fn=lambda result, ag: (
            "normalized" in result.lower() and
            ag.registry.load_dataframe("products_clean")["Price"].between(0, 1).all()
        ),
        agent=agent,
    )

    run_test(
        label="Normalize non-numeric column → blocked",
        query="normalize category in products",
        check_fn=lambda result, ag: "not numeric" in result.lower(),
        agent=agent,
    )

    restore(); backup()

    registry, agent = fresh_agent()

    run_test(
        label="Encode Category → integer codes",
        query="encode category in products",
        check_fn=lambda result, ag: (
            "label-encoded" in result.lower() and
            pd.api.types.is_integer_dtype(
                ag.registry.load_dataframe("products_clean")["Category"]
            )
        ),
        agent=agent,
    )

    run_test(
        label="Encode numeric column → blocked",
        query="encode price in products",
        check_fn=lambda result, ag: "not categorical" in result.lower(),
        agent=agent,
    )

    restore(); backup()

    registry, agent = fresh_agent()

    run_test(
        label="Rename Stock to inventory",
        query="rename stock to inventory in products",
        check_fn=lambda result, ag: (
            "renamed" in result.lower() and
            "inventory" in ag.registry.load_dataframe("products_clean").columns and
            "Stock" not in ag.registry.load_dataframe("products_clean").columns
        ),
        agent=agent,
    )

    restore(); backup()

    # ── EDGE CASES ──────────────────────────────────────────
    print("--- Edge Cases ---\n")

    registry, agent = fresh_agent()

    run_test(
        label="Unknown operation → fallback message",
        query="sort price in products",
        check_fn=lambda result, ag: "not understood" in result.lower(),
        agent=agent,
    )

finally:
    restore()

print("=" * 60)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed.")
print("=" * 60)
