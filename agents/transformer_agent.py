from data.schema_extractor import extract_schema
from utils.logger import logger


class TransformerAgent:
    """
    All operations target a '<dataset>_clean' copy of the original.
    The source dataset is never modified.

    Cleaning (primary):
        drop duplicates in <dataset>
        fill nulls in <dataset>              — fills all columns
        fill nulls in <col> in <dataset>     — fills one column
        drop column <col> in <dataset>
        drop constant columns in <dataset>
        strip whitespace in <dataset>

    Transformation (secondary — run after cleaning):
        normalize <col> in <dataset>
        encode <col> in <dataset>
        rename <old> to <new> in <dataset>
    """

    def __init__(self, registry):
        self.registry = registry


    # ── helpers ────────────────────────────────────────────────────────────

    def _detect_source(self, query, datasets):
        """
        Match the base dataset name from the query.
        Strips the _clean suffix so users can reference either form.
        """
        q = query.lower()
        base_datasets = [d for d in datasets if not d.endswith("_clean")]
        for d in base_datasets:
            if d.lower() in q:
                return d
        return base_datasets[0] if base_datasets else datasets[0]


    def _detect_column(self, query, columns):
        q = query.lower()
        for col in columns:
            if col.lower() in q:
                return col
        return None


    def _get_working_dataset(self, source_name):
        """
        Returns (clean_name, df) — creating the _clean copy if needed.
        """
        clean_name = f"{source_name}_clean"
        datasets = self.registry.list_datasets()

        if clean_name not in datasets:
            source_df = self.registry.load_dataframe(source_name)
            schema = extract_schema(source_df)
            self.registry.register_dataset(clean_name, source_df.copy(), schema)
            logger.info(f"Created clean copy: {clean_name}")

        df = self.registry.load_dataframe(clean_name)
        return clean_name, df


    def _save(self, name, df):
        schema = extract_schema(df)
        self.registry.update_dataset(name, df, schema)


    def _smart_fill_column(self, series):
        """
        Numeric: use mean when |skewness| < 1, median otherwise.
        Categorical / string: use mode.
        Returns (filled_series, fill_label).
        """
        dtype_name = series.dtype.name

        if dtype_name in ("int64", "float64", "int32", "float32"):
            skewness = abs(series.skew())
            if skewness < 1:
                value = series.mean()
                return series.fillna(value), f"mean ({round(value, 4)})"
            else:
                value = series.median()
                return series.fillna(value), f"median ({round(value, 4)})"

        mode_val = series.mode()
        if mode_val.empty:
            return series, "no mode found"
        value = mode_val[0]
        return series.fillna(value), f"mode ('{value}')"


    # ── main handler ───────────────────────────────────────────────────────

    def handle(self, query):

        q = query.lower()

        try:
            all_datasets = self.registry.list_datasets()

            if not all_datasets:
                logger.warning("TransformerAgent called with no datasets loaded.")
                return "No datasets available."

            source = self._detect_source(q, all_datasets)
            clean_name, df = self._get_working_dataset(source)
            columns = df.columns.tolist()

        except Exception as e:
            logger.error(f"TransformerAgent failed to load dataset | {e}")
            return "Failed to load dataset."


        try:

            # ── CLEANING OPERATIONS ─────────────────────────────────────

            # DROP DUPLICATES
            if "duplicate" in q:
                before = len(df)
                str_cols = [c for c in df.columns if df[c].dtype.name in ("str", "string", "object")]
                df[str_cols] = df[str_cols].astype(object)
                df = df.drop_duplicates()
                removed = before - len(df)
                self._save(clean_name, df)
                logger.info(f"Dropped {removed} duplicates from {clean_name}")
                return (
                    f"Dropped {removed} duplicate row(s) from '{clean_name}'. "
                    f"Rows remaining: {len(df)}."
                )

            # DROP CONSTANT COLUMNS
            if "constant" in q:
                constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
                if not constant_cols:
                    return f"No constant columns found in '{clean_name}'."
                df = df.drop(columns=constant_cols)
                self._save(clean_name, df)
                logger.info(f"Dropped constant columns {constant_cols} from {clean_name}")
                return (
                    f"Dropped {len(constant_cols)} constant column(s) from '{clean_name}': "
                    f"{', '.join(constant_cols)}."
                )

            # STRIP WHITESPACE
            if "strip" in q or "whitespace" in q:
                string_cols = [
                    c for c in df.columns
                    if df[c].dtype.name in ("object", "str", "string")
                ]
                if not string_cols:
                    return f"No string columns found in '{clean_name}' to strip."
                for col in string_cols:
                    df[col] = df[col].str.strip()
                self._save(clean_name, df)
                logger.info(f"Stripped whitespace in {len(string_cols)} columns in {clean_name}")
                return (
                    f"Stripped leading/trailing whitespace from {len(string_cols)} "
                    f"string column(s) in '{clean_name}'."
                )

            # FILL NULLS
            if "fill" in q or "impute" in q:
                column = self._detect_column(q, columns)

                # single column
                if column is not None:
                    null_count = df[column].isnull().sum()
                    if null_count == 0:
                        return f"Column '{column}' in '{clean_name}' has no missing values."
                    df[column], label = self._smart_fill_column(df[column])
                    self._save(clean_name, df)
                    logger.info(f"Filled {null_count} nulls in '{column}' in {clean_name}")
                    return (
                        f"Filled {null_count} missing value(s) in '{column}' "
                        f"using {label}."
                    )

                # all columns
                report = []
                for col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        df[col], label = self._smart_fill_column(df[col])
                        report.append(f"  '{col}': {null_count} filled using {label}")

                if not report:
                    return f"No missing values found in '{clean_name}'."

                self._save(clean_name, df)
                logger.info(f"Filled nulls across all columns in {clean_name}")
                return "Filled missing values:\n" + "\n".join(report)

            # DROP COLUMN
            if "drop" in q:
                column = self._detect_column(q, columns)
                if column is None:
                    return "Column not found in dataset."
                df = df.drop(columns=[column])
                self._save(clean_name, df)
                logger.info(f"Dropped column '{column}' from {clean_name}")
                return f"Column '{column}' dropped from '{clean_name}'."


            # ── TRANSFORMATION OPERATIONS ───────────────────────────────

            # NORMALIZE
            if "normalize" in q or "scale" in q:
                column = self._detect_column(q, columns)
                if column is None:
                    return "Column not found in dataset."

                if df[column].dtype.name not in ("int64", "float64", "int32", "float32"):
                    return f"Column '{column}' is not numeric. Cannot normalize."

                col_min, col_max = df[column].min(), df[column].max()
                if col_max == col_min:
                    return f"Column '{column}' has a constant value. Cannot normalize."

                df[column] = (df[column] - col_min) / (col_max - col_min)
                self._save(clean_name, df)
                logger.info(f"Normalized '{column}' in {clean_name}")
                return f"Column '{column}' in '{clean_name}' normalized to [0, 1]."

            # ENCODE
            if "encode" in q:
                column = self._detect_column(q, columns)
                if column is None:
                    return "Column not found in dataset."

                dtype_name = df[column].dtype.name
                if dtype_name not in ("object", "category", "str", "string"):
                    return f"Column '{column}' is not categorical. Cannot encode."

                categories = df[column].astype("category").cat.categories.tolist()
                df[column] = df[column].astype("category").cat.codes
                self._save(clean_name, df)
                logger.info(f"Label-encoded '{column}' in {clean_name}")
                return (
                    f"Column '{column}' in '{clean_name}' label-encoded. "
                    f"Categories: {categories[:10]}"
                    f"{'...' if len(categories) > 10 else ''}"
                )

            # RENAME
            if "rename" in q and " to " in q:
                try:
                    after = q.split("rename", 1)[1]
                    parts = after.split(" to ", 1)
                    old_raw = parts[0].strip()
                    new_raw = parts[1].strip().split()[0]
                    old_name = next((c for c in columns if c.lower() == old_raw), None)
                    if old_name is None:
                        return f"Column '{old_raw}' not found in dataset."
                    df = df.rename(columns={old_name: new_raw})
                    self._save(clean_name, df)
                    logger.info(f"Renamed '{old_name}' → '{new_raw}' in {clean_name}")
                    return f"Column '{old_name}' renamed to '{new_raw}' in '{clean_name}'."
                except Exception:
                    return "Could not parse rename. Use: rename <old> to <new> in <dataset>"

            return (
                "Operation not understood. Supported — "
                "cleaning: drop duplicates, fill nulls, drop column, drop constant columns, strip whitespace; "
                "transforms: normalize, encode, rename."
            )

        except Exception as e:
            logger.error(f"TransformerAgent error | Query: {query} | {e}")
            return "Transformer agent error."
