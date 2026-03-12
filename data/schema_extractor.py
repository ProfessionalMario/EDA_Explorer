import pandas as pd


def extract_schema(df: pd.DataFrame):

    schema = {}

    schema["rows"] = len(df)

    schema["columns"] = list(df.columns)

    schema["numeric_columns"] = list(
        df.select_dtypes(include=["number"]).columns
    )

    schema["categorical_columns"] = list(
        df.select_dtypes(include=["object", "category"]).columns
    )

    schema["missing_values"] = (
        df.isnull().mean().round(4).to_dict()
    )

    # NEW FIELD
    schema["column_types"] = {
        col: str(dtype) for col, dtype in df.dtypes.items()
    }

    return schema