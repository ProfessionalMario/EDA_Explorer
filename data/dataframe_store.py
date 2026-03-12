class DataFrameStore:
    """
    A simple in-memory manager for storing and accessing multiple datasets.

    Each dataset is stored with:
    - original: the untouched DataFrame
    - working: a copy used for transformations
    - schema: metadata describing the dataset structure
    """

    def __init__(self):
        """Initialize an empty dataset store."""
        self.datasets = {}

    def add_dataset(self, name, df, schema):
        """
        Add a dataset to the store.

        Parameters
        ----------
        name : str
            Unique name used to identify the dataset.
        df : pandas.DataFrame
            The DataFrame to store.
        schema : dict
            Metadata describing column types or structure.
        """
        if name in self.datasets:
            raise ValueError(f"Dataset '{name}' already loaded")

        self.datasets[name] = {
            "original": df,
            "working": df.copy(),
            "schema": schema
        }

    def list_datasets(self):
        """
        Return a list of all dataset names currently stored.
        """
        return list(self.datasets.keys())

    def get_dataset(self, name):
        """
        Retrieve the dataset dictionary for a given dataset name.

        Returns
        -------
        dict
            Contains 'original', 'working', and 'schema'.
        """
        return self.datasets.get(name)

    def get_schema(self, name):
        """
        Get the schema metadata for a specific dataset.
        """
        return self.datasets[name]["schema"]