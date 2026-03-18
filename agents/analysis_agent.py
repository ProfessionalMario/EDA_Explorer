from utils.logger import logger
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class AnalysisAgent:

    def __init__(self, registry):
        
        self.registry = registry

    # ---------------------------
    # Proper dataset extraction
    # ---------------------------
    def _extract_dataset(self, text):
        print("🔍 Running dataset analysis...")
        datasets = self.registry.list_datasets()
        words = str(text).lower().split()

        for word in words:
            for d in datasets:
                if word == d.lower():
                    return d
        return None

    # ---------------------------
    # Remove ID-like columns
    # ---------------------------
    def _remove_id_like_columns(self, df):
        cols_to_drop = []

        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)

            if unique_ratio > 0.9:
                cols_to_drop.append(col)

        df_clean = df.drop(columns=cols_to_drop)

        return df_clean, cols_to_drop

    # ---------------------------
    # Select target column
    # ---------------------------
    def _select_target(self, df):
        candidates = []

        for col in df.columns:
            unique_count = df[col].nunique()
            unique_ratio = unique_count / len(df)

            # Skip obvious bad columns
            if any(k in col.lower() for k in ["id", "name", "email", "phone"]):
                continue

            #  Skip high-cardinality
            if unique_ratio > 0.5:
                continue

            # Prefer categorical / classification targets
            if unique_count <= 20:
                candidates.append((col, unique_count))

        #  Pick best candidate (lowest unique count but >1)
        if candidates:
            candidates = sorted(candidates, key=lambda x: x[1])
            return candidates[0][0]

        return None

    # ---------------------------
    # Feature importance
    # ---------------------------
    def _compute_feature_importance(self, df):

        df_clean, dropped_cols = self._remove_id_like_columns(df)
        if len(df.columns) <= 2:
            return None, dropped_cols, "Dataset too small for feature importance."
        target = self._select_target(df_clean)
        if not target:
            return None, dropped_cols, "No suitable target column found."
        
        y = df_clean[target]

        # Fix NaN issue
        if y.isnull().sum() > 0:
            return None, dropped_cols, "Target contains missing values. Cannot compute feature importance."
        if not target:
            return None, dropped_cols, "No suitable target column found."

        #  Prevent sklearn warning
        if df_clean[target].nunique() > 0.5 * len(df_clean):
            return None, dropped_cols, "Target not suitable for classification."

        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        # Encode categoricals
        X = X.apply(lambda col: col.astype('category').cat.codes)

        try:
            model = RandomForestClassifier(n_estimators=50)
            model.fit(X, y)

            importances = dict(zip(X.columns, model.feature_importances_))
            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

            return sorted_imp[:5], dropped_cols, None

        except Exception as e:
            return None, dropped_cols, str(e)

    # ---------------------------
    # Optional explanation layer
    # ---------------------------
    def _explain_feature(self, col):
        return f"{col} shows strong predictive signal based on dataset patterns."
    
    #----------------------------
    # Outlier Detection
    #----------------------------
    def _detect_outliers(self, df):
        try:
            numeric_df = df.select_dtypes(include="number")

            outlier_summary = {}

            for col in numeric_df.columns:
                q1 = numeric_df[col].quantile(0.25)
                q3 = numeric_df[col].quantile(0.75)
                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                outliers = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)]

                if len(outliers) > 0:
                    outlier_summary[col] = len(outliers)

            return outlier_summary, None

        except Exception as e:
            logger.error(f"Outlier detection failed | {e}")
            return None, str(e)
    
    #---------------------------
    # Correlation analysis
    #---------------------------
    def _compute_correlation(self, df):
        try:
            numeric_df = df.select_dtypes(include="number")

            if numeric_df.shape[1] < 2:
                return None, "Not enough numeric columns for correlation."

            corr = numeric_df.corr()

            # Get top correlations (excluding self)
            corr_matrix = numeric_df.corr().abs()

            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            top_pairs = (
                upper.unstack()
                .dropna()
                .sort_values(ascending=False)
                .head(5)
            )
            return top_pairs.to_dict(), None

        except Exception as e:
            logger.error(f"Correlation failed | {e}")
            return None, str(e)
        
    #-------------------------
    #Saving report    
    #-------------------------
    def _export_report(self, dataset, content):
        try:
            path = f"output/report_{dataset}.txt"

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Report exported: {path}")

            return path

        except Exception as e:
            logger.error(f"Report export failed | {e}")
            return None

    # ---------------------------
    # MAIN HANDLER
    # ---------------------------
    def handle(self, dataset=None):

        try:
            # ---- HANDLE "analyze people" CASE ----
            if isinstance(dataset, str):
                extracted = self._extract_dataset(dataset)
                if extracted:
                    dataset = extracted

            # ---- STRICT DATASET CHECK ----
            if not dataset:
                return "Please specify a dataset (e.g., 'analyze people')"

            df = self.registry.load_dataframe(dataset)

        except Exception as e:
            logger.error(f"Failed loading dataset | {e}")
            return f"Failed to load dataset: {dataset}"

        try:
            # ---------- OUTPUT ----------
            output = []
            rows, cols = df.shape
            print("🧹 Checking duplicates...")
            # ---------- DATA QUALITY ----------
            total_missing = df.isnull().sum().sum()
            duplicates = df.duplicated().sum()

            missing_by_column = df.isnull().sum()
            missing_by_column = missing_by_column[missing_by_column > 0]

            # ---------- COLUMN TYPES ----------
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

            # ---------- WARNINGS ----------
            print("⚠️ Generating warnings...")
            warnings = []

            for col in df.columns:
                if len(df) == 0:
                    continue

                unique_ratio = df[col].nunique() / len(df)

                if unique_ratio > 0.95 and "id" in col.lower():
                    warnings.append(f"{col} looks like an ID column")

                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio > 0.5:
                    warnings.append(f"{col} has {missing_ratio:.2%} missing values")

                if df[col].nunique() == 1:
                    warnings.append(f"{col} is constant (no variance)")

            # ---------- FEATURE IMPORTANCE (NEW CLEAN VERSION) ----------
            print("📈 Looking for potential feature importance...")
            fi, dropped_cols, error = self._compute_feature_importance(df)

            # ---------- CORRELATION ANALYSIS ----------
            print("📊 Computing correlation...")
            corr_pairs, corr_error = self._compute_correlation(df)

            # ---------- EXPORT (ONLY ONCE) ----------
            report_path = self._export_report(dataset, "\n".join(output))

            if report_path:
                output.append(f"\n📁 Report saved to: {report_path}")
            

            output.append(f"\nDataset Analysis: {dataset}")
            output.append("=" * 40)

            output.append(f"Rows: {rows}")
            output.append(f"Columns: {cols}")

            output.append("\nData Quality")
            output.append("-" * 20)
            output.append(f"Total Missing Values : {total_missing}")
            output.append(f"Duplicate Rows       : {duplicates}")
            # ---------- CORRELATION OUTPUT ----------
            output.append("\nTop Correlations")
            output.append("-" * 20)

            if corr_error:
                output.append(corr_error)
            elif corr_pairs is not None:
                for (col1, col2), val in corr_pairs.items():
                    output.append(f"{col1} ↔ {col2}: {val:.3f}")
            else:
                output.append("No correlation data available.")
            if not missing_by_column.empty:
                output.append("\nMissing by Column")
                output.append("-" * 20)
                for col, val in missing_by_column.items():
                    output.append(f"{col}: {val}")

            output.append("\nColumn Types")
            output.append("-" * 20)
            output.append(f"Numeric      : {', '.join(numeric_cols) if numeric_cols else 'None'}")
            output.append(f"Categorical  : {', '.join(categorical_cols) if categorical_cols else 'None'}")

            if warnings:
                output.append("\n⚠️ Data Warnings")
                output.append("-" * 20)
                for w in warnings[:5]:
                    output.append(f"- {w}")

            # ---------- FEATURE IMPORTANCE OUTPUT ----------
            output.append("\nPotential Feature Importance")
            output.append("-" * 20)

            if error:
                output.append(error)
            else:
                for col, score in fi:
                    explanation = self._explain_feature(col)
                    output.append(f"{col}: {score:.4f} → {explanation}")

            # ---------- DROPPED COLUMNS ----------
            if dropped_cols:
                output.append("\n⚠️ Ignored high-cardinality columns:")
                for col in dropped_cols:
                    output.append(f"- {col}")

            return "\n".join(output)

        except Exception as e:
            logger.error(f"Analysis failed | {e}")
            return "Analysis agent error."