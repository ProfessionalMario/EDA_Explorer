import pandas as pd
from data.schema_extractor import extract_schema
from utils.logger import logger


NUMERIC_TYPES = ("int64", "float64", "int32", "float32")
CATEGORICAL_TYPES = ("object", "category", "str", "string")


class TransformerAgent:
    """
    All operations target a '<dataset>_clean' copy of the original.
    The source dataset is never modified.

    Supports plan-based dispatch (from LLM) and keyword-based fallback.

    Cleaning:
        drop_duplicates, drop_column, drop_constant_columns,
        strip_whitespace, drop_missing_rows, drop_missing_cols

    Filling:
        fill_nulls  — smart auto (mean/median for numeric, mode for categorical)
        fill_mean   — explicit mean
        fill_median — explicit median
        fill_mode   — explicit mode
        fill_zero   — fill with 0 / empty string

    Scaling / Encoding:
        normalize   — min-max [0, 1]
        standardize — z-score (subtract mean, divide by std)
        encode      — label encoding (int codes)
        onehot      — one-hot encoding (pd.get_dummies, ≤20 unique values)

    Other:
        rename      — rename <old> to <new> in <dataset>
    """

    def __init__(self, registry):
        self.registry = registry

    # ── helpers ────────────────────────────────────────────────────────────

    def _detect_source(self, query, datasets):
        q = query.lower()
        base = [d for d in datasets if not d.endswith("_clean")]
        for d in base:
            if d.lower() in q:
                return d
        return base[0] if base else datasets[0]

    def _detect_column(self, query, columns):
        q = query.lower()
        for col in columns:
            if col.lower() in q:
                return col
        return None

    def _resolve_column(self, plan_col, query, columns):
        """
        Return the real column name, checking the plan first then falling
        back to keyword scan. Returns None if nothing found.
        """
        if plan_col:
            for col in columns:
                if col.lower() == plan_col.lower():
                    return col
        return self._detect_column(query, columns)

    def _get_working_dataset(self, source_name):
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
        self.registry.update_dataset(name, df, extract_schema(df))

    def _smart_fill_column(self, series):
        """Auto pick mean/median for numeric, mode for categorical."""
        if series.dtype.name in NUMERIC_TYPES:
            skewness = abs(series.skew())
            if skewness < 1:
                val = series.mean()
                return series.fillna(val), f"mean ({round(val, 4)})"
            val = series.median()
            return series.fillna(val), f"median ({round(val, 4)})"
        mode_val = series.mode()
        if mode_val.empty:
            return series, "no mode found"
        val = mode_val[0]
        return series.fillna(val), f"mode ('{val}')"

    # ── individual operation methods ───────────────────────────────────────

    def _op_drop_duplicates(self, clean_name, df):
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

    def _op_drop_constant_columns(self, clean_name, df):
        const_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if not const_cols:
            return f"No constant columns found in '{clean_name}'."
        df = df.drop(columns=const_cols)
        self._save(clean_name, df)
        logger.info(f"Dropped constant columns {const_cols} from {clean_name}")
        return (
            f"Dropped {len(const_cols)} constant column(s) from '{clean_name}': "
            f"{', '.join(const_cols)}."
        )

    def _op_strip_whitespace(self, clean_name, df):
        str_cols = [c for c in df.columns if df[c].dtype.name in ("object", "str", "string")]
        if not str_cols:
            return f"No string columns to strip in '{clean_name}'."
        for col in str_cols:
            df[col] = df[col].str.strip()
        self._save(clean_name, df)
        logger.info(f"Stripped whitespace in {len(str_cols)} columns in {clean_name}")
        return (
            f"Stripped leading/trailing whitespace from {len(str_cols)} "
            f"string column(s) in '{clean_name}'."
        )

    def _op_drop_missing_rows(self, clean_name, df):
        before = len(df)
        df = df.dropna()
        removed = before - len(df)
        self._save(clean_name, df)
        logger.info(f"Dropped {removed} rows with nulls from {clean_name}")
        return (
            f"Dropped {removed} row(s) containing missing values from '{clean_name}'. "
            f"Rows remaining: {len(df)}."
        )

    def _op_drop_missing_cols(self, clean_name, df, threshold=0.5):
        missing_pct = df.isnull().mean()
        drop_cols = missing_pct[missing_pct > threshold].index.tolist()
        if not drop_cols:
            return f"No columns exceed {int(threshold * 100)}% missing threshold in '{clean_name}'."
        df = df.drop(columns=drop_cols)
        self._save(clean_name, df)
        logger.info(f"Dropped high-null columns {drop_cols} from {clean_name}")
        return (
            f"Dropped {len(drop_cols)} column(s) with >{int(threshold * 100)}% missing "
            f"from '{clean_name}': {', '.join(drop_cols)}."
        )

    def _op_drop_column(self, clean_name, df, column):
        if column is None:
            return "Column not found in dataset."
        df = df.drop(columns=[column])
        self._save(clean_name, df)
        logger.info(f"Dropped column '{column}' from {clean_name}")
        return f"Column '{column}' dropped from '{clean_name}'."

    def _op_fill_smart(self, clean_name, df, column):
        """Smart fill — auto selects mean/median/mode."""
        if column is not None:
            nulls = df[column].isnull().sum()
            if nulls == 0:
                return f"Column '{column}' in '{clean_name}' has no missing values."
            df[column], label = self._smart_fill_column(df[column])
            self._save(clean_name, df)
            logger.info(f"Smart-filled {nulls} nulls in '{column}' in {clean_name}")
            return f"Filled {nulls} missing value(s) in '{column}' using {label}."

        report = []
        for col in df.columns:
            nulls = df[col].isnull().sum()
            if nulls > 0:
                df[col], label = self._smart_fill_column(df[col])
                report.append(f"  '{col}': {nulls} filled using {label}")
        if not report:
            return f"No missing values found in '{clean_name}'."
        self._save(clean_name, df)
        logger.info(f"Smart-filled all nulls in {clean_name}")
        return "Filled missing values:\n" + "\n".join(report)

    def _op_fill_with(self, clean_name, df, column, strategy):
        """
        Fill with an explicit strategy: mean | median | mode | zero.
        If column is None, applies to all eligible columns.
        """
        if strategy == "zero":
            targets = [column] if column else df.columns.tolist()
            report = []
            for col in targets:
                nulls = df[col].isnull().sum()
                if nulls == 0:
                    continue
                fill_val = 0 if df[col].dtype.name in NUMERIC_TYPES else ""
                df[col] = df[col].fillna(fill_val)
                report.append(f"  '{col}': {nulls} filled with {fill_val!r}")
            if not report:
                return f"No missing values found in '{clean_name}'."
            self._save(clean_name, df)
            return f"Filled with zero:\n" + "\n".join(report)

        if strategy in ("mean", "median"):
            targets = (
                [column] if column
                else [c for c in df.columns if df[c].dtype.name in NUMERIC_TYPES]
            )
            if not targets:
                return "No numeric columns available for this operation."
            report = []
            for col in targets:
                nulls = df[col].isnull().sum()
                if nulls == 0:
                    continue
                if df[col].dtype.name not in NUMERIC_TYPES:
                    report.append(f"  '{col}': skipped (not numeric)")
                    continue
                val = df[col].mean() if strategy == "mean" else df[col].median()
                df[col] = df[col].fillna(val)
                report.append(f"  '{col}': {nulls} filled with {strategy} ({round(val, 4)})")
            if not report:
                return f"No missing values in numeric columns of '{clean_name}'."
            self._save(clean_name, df)
            return f"Filled with {strategy}:\n" + "\n".join(report)

        if strategy == "mode":
            targets = [column] if column else df.columns.tolist()
            report = []
            for col in targets:
                nulls = df[col].isnull().sum()
                if nulls == 0:
                    continue
                mode_val = df[col].mode()
                if mode_val.empty:
                    report.append(f"  '{col}': skipped (no mode)")
                    continue
                df[col] = df[col].fillna(mode_val[0])
                report.append(f"  '{col}': {nulls} filled with mode ('{mode_val[0]}')")
            if not report:
                return f"No missing values found in '{clean_name}'."
            self._save(clean_name, df)
            return f"Filled with mode:\n" + "\n".join(report)

        return f"Unknown fill strategy: {strategy!r}"

    def _op_normalize(self, clean_name, df, column, columns):
        """Min-max normalization to [0, 1]."""
        targets = (
            [column] if column
            else [c for c in columns if df[c].dtype.name in NUMERIC_TYPES]
        )
        if not targets:
            return "No numeric columns to normalize."
        report = []
        for col in targets:
            if df[col].dtype.name not in NUMERIC_TYPES:
                report.append(f"  '{col}': skipped (not numeric)")
                continue
            col_min, col_max = df[col].min(), df[col].max()
            if col_max == col_min:
                report.append(f"  '{col}': skipped (constant value)")
                continue
            df[col] = (df[col] - col_min) / (col_max - col_min)
            report.append(f"  '{col}': normalized to [0, 1]")
        if not report:
            return f"No columns were normalized in '{clean_name}'."
        self._save(clean_name, df)
        logger.info(f"Normalized columns in {clean_name}")
        return f"Min-max normalization applied in '{clean_name}':\n" + "\n".join(report)

    def _op_standardize(self, clean_name, df, column, columns):
        """Z-score standardization: (x - mean) / std."""
        targets = (
            [column] if column
            else [c for c in columns if df[c].dtype.name in NUMERIC_TYPES]
        )
        if not targets:
            return "No numeric columns to standardize."
        report = []
        for col in targets:
            if df[col].dtype.name not in NUMERIC_TYPES:
                report.append(f"  '{col}': skipped (not numeric)")
                continue
            std = df[col].std()
            if std == 0:
                report.append(f"  '{col}': skipped (zero variance)")
                continue
            mean = df[col].mean()
            df[col] = (df[col] - mean) / std
            report.append(
                f"  '{col}': standardized (mean={round(mean, 4)}, std={round(std, 4)})"
            )
        if not report:
            return f"No columns were standardized in '{clean_name}'."
        self._save(clean_name, df)
        logger.info(f"Standardized columns in {clean_name}")
        return f"Z-score standardization applied in '{clean_name}':\n" + "\n".join(report)

    def _op_encode(self, clean_name, df, column):
        """Label encoding — categorical → integer codes."""
        if column is None:
            return "Specify a column to encode."
        if df[column].dtype.name not in CATEGORICAL_TYPES:
            return f"Column '{column}' is not categorical. Cannot label-encode."
        categories = df[column].astype("category").cat.categories.tolist()
        df[column] = df[column].astype("category").cat.codes
        self._save(clean_name, df)
        logger.info(f"Label-encoded '{column}' in {clean_name}")
        return (
            f"Column '{column}' in '{clean_name}' label-encoded. "
            f"Categories: {categories[:10]}"
            f"{'...' if len(categories) > 10 else ''}"
        )

    def _op_onehot(self, clean_name, df, column, columns, max_unique=20):
        """One-hot encoding via pd.get_dummies (≤max_unique unique values)."""
        targets = (
            [column] if column
            else [c for c in columns if df[c].dtype.name in CATEGORICAL_TYPES]
        )
        if not targets:
            return "No categorical columns for one-hot encoding."
        report = []
        for col in targets:
            if df[col].dtype.name not in CATEGORICAL_TYPES:
                report.append(f"  '{col}': skipped (not categorical)")
                continue
            unique_n = df[col].nunique()
            if unique_n > max_unique:
                report.append(
                    f"  '{col}': skipped ({unique_n} unique values exceeds limit of {max_unique})"
                )
                continue
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)
            report.append(f"  '{col}': expanded into {len(dummies.columns)} columns")
        if not report:
            return f"No columns were one-hot encoded in '{clean_name}'."
        self._save(clean_name, df)
        logger.info(f"One-hot encoded columns in {clean_name}")
        return f"One-hot encoding applied in '{clean_name}':\n" + "\n".join(report)

    def _op_rename(self, clean_name, df, columns, query):
        q = query.lower()
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

    # ── plan-based dispatch ────────────────────────────────────────────────

    def _dispatch_plan(self, plan, clean_name, df, columns):
        """
        Directly execute the operation named in the LLM plan.
        Bypasses keyword matching for precise execution.
        """
        op     = plan.get("operation")
        p_col  = plan.get("column")
        column = self._resolve_column(p_col, "", columns)

        dispatch = {
            "drop_duplicates":      lambda: self._op_drop_duplicates(clean_name, df),
            "drop_constant_columns":lambda: self._op_drop_constant_columns(clean_name, df),
            "strip_whitespace":     lambda: self._op_strip_whitespace(clean_name, df),
            "drop_missing_rows":    lambda: self._op_drop_missing_rows(clean_name, df),
            "drop_missing_cols":    lambda: self._op_drop_missing_cols(clean_name, df),
            "drop_column":          lambda: self._op_drop_column(clean_name, df, column),
            "fill_nulls":           lambda: self._op_fill_smart(clean_name, df, column),
            "fill_mean":            lambda: self._op_fill_with(clean_name, df, column, "mean"),
            "fill_median":          lambda: self._op_fill_with(clean_name, df, column, "median"),
            "fill_mode":            lambda: self._op_fill_with(clean_name, df, column, "mode"),
            "fill_zero":            lambda: self._op_fill_with(clean_name, df, column, "zero"),
            "normalize":            lambda: self._op_normalize(clean_name, df, column, columns),
            "standardize":          lambda: self._op_standardize(clean_name, df, column, columns),
            "encode":               lambda: self._op_encode(clean_name, df, column),
            "onehot":               lambda: self._op_onehot(clean_name, df, column, columns),
            "rename":               lambda: self._op_rename(clean_name, df, columns, ""),
        }

        fn = dispatch.get(op)
        if fn:
            logger.info(f"Plan dispatch | op={op} | col={column} | dataset={clean_name}")
            return fn()
        return (
            f"Operation '{op}' is not implemented in the transformer agent."
        )

    # ── keyword-based fallback ─────────────────────────────────────────────

    def _dispatch_keywords(self, q, query, clean_name, df, columns):
        """Keyword-based routing used when no LLM plan is available."""

        # ── CLEANING ────────────────────────────────────────────────────

        if "duplicate" in q:
            return self._op_drop_duplicates(clean_name, df)

        if "constant" in q:
            return self._op_drop_constant_columns(clean_name, df)

        if "strip" in q or "whitespace" in q:
            return self._op_strip_whitespace(clean_name, df)

        if "drop missing row" in q or "drop na" in q or "dropna" in q:
            return self._op_drop_missing_rows(clean_name, df)

        if "drop missing col" in q:
            return self._op_drop_missing_cols(clean_name, df)

        # ── FILLING ─────────────────────────────────────────────────────

        if "fill" in q or "impute" in q:
            column = self._detect_column(q, columns)

            # Explicit strategy keywords take priority over smart fill
            if "mean" in q:
                return self._op_fill_with(clean_name, df, column, "mean")
            if "median" in q:
                return self._op_fill_with(clean_name, df, column, "median")
            if "mode" in q:
                return self._op_fill_with(clean_name, df, column, "mode")
            if "zero" in q or " 0 " in q:
                return self._op_fill_with(clean_name, df, column, "zero")

            # Default: smart auto-fill
            return self._op_fill_smart(clean_name, df, column)

        # ── DROP COLUMN ─────────────────────────────────────────────────

        if "drop" in q:
            column = self._detect_column(q, columns)
            return self._op_drop_column(clean_name, df, column)

        # ── TRANSFORMS ──────────────────────────────────────────────────

        if "standardize" in q or "zscore" in q or "z-score" in q:
            column = self._detect_column(q, columns)
            return self._op_standardize(clean_name, df, column, columns)

        if "normalize" in q or "scale" in q:
            column = self._detect_column(q, columns)
            return self._op_normalize(clean_name, df, column, columns)

        if "one hot" in q or "onehot" in q or "one-hot" in q or "dummies" in q:
            column = self._detect_column(q, columns)
            return self._op_onehot(clean_name, df, column, columns)

        if "encode" in q:
            column = self._detect_column(q, columns)
            return self._op_encode(clean_name, df, column)

        if "rename" in q and " to " in q:
            return self._op_rename(clean_name, df, columns, query)

        return (
            "Operation not understood. Supported — "
            "cleaning: drop duplicates, drop column, drop constant columns, "
            "strip whitespace, drop missing rows, drop missing cols; "
            "filling: fill nulls / fill with mean / median / mode / zero; "
            "scaling: normalize, standardize; "
            "encoding: encode (label), onehot; "
            "other: rename."
        )

    # ── public entry point ─────────────────────────────────────────────────

    def handle(self, query, plan=None):
        q = query.lower()

        try:
            all_datasets = self.registry.list_datasets()
            if not all_datasets:
                logger.warning("TransformerAgent called with no datasets loaded.")
                return "No datasets available."

            # Dataset resolution: prefer plan's dataset, fall back to keyword scan
            if plan and plan.get("dataset"):
                raw_ds = plan["dataset"].replace("_clean", "")
                source = raw_ds if raw_ds in all_datasets else self._detect_source(q, all_datasets)
            else:
                source = self._detect_source(q, all_datasets)

            clean_name, df = self._get_working_dataset(source)
            columns = df.columns.tolist()

        except Exception as e:
            logger.error(f"TransformerAgent failed to load dataset | {e}")
            return "Failed to load dataset."

        try:
            if plan and plan.get("operation"):
                return self._dispatch_plan(plan, clean_name, df, columns)
            return self._dispatch_keywords(q, query, clean_name, df, columns)

        except Exception as e:
            logger.error(f"TransformerAgent error | Query: {query} | {e}")
            return "Transformer agent error."
