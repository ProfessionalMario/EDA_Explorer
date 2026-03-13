
DataFrameAgent

Purpose
- Performs basic analytical operations on loaded pandas DataFrames.

Current Capabilities
- Detect dataset name from query
- Detect column name from query
- Extract numeric values (for row limits)
- Execute simple pandas statistical operations

Supported Operations
- show top/first N rows
- row count
- mean / average of a column
- max / highest value of a column
- min / lowest value of a column

Examples
- show top 5 rows in products
- how many rows in customers
- average price in products
- max price in products
- min price in products

Limitations
- rule-based parsing only
- requires exact column names
- limited natural language understanding
- no sorting, filtering, or aggregations yet
- no fuzzy matching or synonym handling

Future Improvements
- LLM planner for query normalization
- fuzzy column matching
- advanced operations (sort, filter, groupby)












