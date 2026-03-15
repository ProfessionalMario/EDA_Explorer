import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
plt.show = lambda: None

import os
from data.registry import DatasetRegistry
from agents.visualization_agent import VisualizationAgent

OUTPUT_DIR = "test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

registry = DatasetRegistry()
agent = VisualizationAgent(registry)

passed = 0
failed = 0


def run_test(label, query, output_file=None, expect_chart=True, expect_message=None):
    global passed, failed
    plt.close("all")

    result = agent.handle(query)
    figures = [plt.figure(i) for i in plt.get_fignums()]

    if expect_chart:
        if figures:
            fig = figures[-1]
            fig.tight_layout()
            path = os.path.join(OUTPUT_DIR, output_file)
            fig.savefig(path, bbox_inches="tight")
            plt.close("all")
            print(f"[PASS] {label}")
            print(f"       Query   : {query}")
            print(f"       Result  : {result}")
            print(f"       Saved to: {path}\n")
            passed += 1
        else:
            print(f"[FAIL] {label}")
            print(f"       Query  : {query}")
            print(f"       Result : {result}")
            print(f"       Expected a chart but none was generated.\n")
            failed += 1
    else:
        plt.close("all")
        ok = (expect_message is None) or (expect_message in result)
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"{tag} {label}")
        print(f"       Query  : {query}")
        print(f"       Result : {result}\n")
        if ok:
            passed += 1
        else:
            failed += 1


print("=" * 60)
print("  Visualization Agent Test Suite")
print("=" * 60)
print(f"Datasets loaded: {registry.list_datasets()}\n")

# ── Histograms (guardrail does NOT apply) ───────────────────
print("--- Histograms ---\n")

run_test(
    label="Histogram – Price (products, 999 unique values, allowed)",
    query="histogram price in products",
    output_file="histogram_price_products.png",
)

run_test(
    label="Histogram – Stock (products, 999 unique values, allowed)",
    query="show histogram of stock in products",
    output_file="histogram_stock_products.png",
)

# ── Bar charts – under limit (guardrail allows) ─────────────
print("--- Bar Charts (within limit) ---\n")

run_test(
    label="Bar chart – Category (products, 34 unique values → allowed)",
    query="bar chart category in products",
    output_file="bar_category_products.png",
)

# ── Bar charts – over limit (guardrail blocks) ──────────────
print("--- Bar Charts (guardrail triggered) ---\n")

run_test(
    label="Bar chart – Color (products, 140 unique values → blocked)",
    query="bar chart color in products",
    expect_chart=False,
    expect_message="Too many to visualize meaningfully",
)

run_test(
    label="Bar chart – Brand (products, 72263 unique values → blocked)",
    query="bar chart brand in products",
    expect_chart=False,
    expect_message="Too many to visualize meaningfully",
)

# ── Edge cases ──────────────────────────────────────────────
print("--- Edge Cases ---\n")

run_test(
    label="No column specified → helpful message",
    query="histogram in products",
    expect_chart=False,
    expect_message="Column not found",
)

run_test(
    label="Unknown dataset → error message",
    query="histogram price in unknown_dataset",
    expect_chart=False,
)

run_test(
    label="Unsupported chart type → fallback message",
    query="scatter plot price in products",
    expect_chart=False,
    expect_message="not understood",
)

# ── Summary ─────────────────────────────────────────────────
print("=" * 60)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed.")
print(f"Charts saved in '{OUTPUT_DIR}/'")
print("=" * 60)
