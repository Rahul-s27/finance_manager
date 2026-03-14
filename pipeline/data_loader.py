"""
Data Loader Module for Finance AutoML Manager.

Handles loading and validation of transaction datasets from various file formats.
"""

import pandas as pd
import os
from typing import Optional, List, Dict, Any


class DataLoader:
    """Load and validate financial transaction datasets."""
    
    REQUIRED_COLUMNS = {"description", "amount", "category"}
    SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls"}
    
    def __init__(self, file_path: str):
        """
        Initialize the DataLoader.
        
        Args:
            file_path: Path to the dataset file
        """
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None
        self._validate_file_path()
    
    def _validate_file_path(self) -> None:
        """Check if file exists and has supported format."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        ext = os.path.splitext(self.file_path)[-1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
    
    def load(self) -> pd.DataFrame:
        """
        Load the dataset from file.
        
        Returns:
            DataFrame containing the loaded data
        """
        ext = os.path.splitext(self.file_path)[-1].lower()
        
        if ext == ".csv":
            self.data = pd.read_csv(self.file_path)
        else:
            self.data = pd.read_excel(self.file_path)
        
        return self.data
    
    def validate_columns(self) -> bool:
        """
        Validate that required columns exist in the dataset.
        
        Returns:
            True if all required columns are present
            
        Raises:
            ValueError: If required columns are missing
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        columns = set(self.data.columns)
        missing = self.REQUIRED_COLUMNS - columns
        
        if missing:
            raise ValueError(
                f"Missing required columns: {', '.join(missing)}. "
                f"Required columns: {', '.join(self.REQUIRED_COLUMNS)}"
            )
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the loaded dataset.
        
        Returns:
            Dictionary containing dataset summary
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        return {
            "rows": len(self.data),
            "columns": list(self.data.columns),
            "category_counts": self.data["category"].value_counts().to_dict() if "category" in self.data.columns else {},
            "amount_stats": {
                "min": float(self.data["amount"].min()) if "amount" in self.data.columns else None,
                "max": float(self.data["amount"].max()) if "amount" in self.data.columns else None,
                "mean": float(self.data["amount"].mean()) if "amount" in self.data.columns else None,
            },
            "missing_values": self.data.isnull().sum().to_dict()
        }
    
    def get_training_data(self) -> pd.DataFrame:
        """
        Get clean training data with required columns only.
        
        Returns:
            DataFrame with description, amount, and category columns
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        self.validate_columns()
        
        # Return only required columns
        return self.data[list(self.REQUIRED_COLUMNS)].copy()


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Convenience function to load and validate a dataset.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Validated DataFrame ready for training
    """
    loader = DataLoader(file_path)
    loader.load()
    loader.validate_columns()
    return loader.get_training_data()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default to training data
        file_path = os.path.join("data", "training_data.csv")
    
    try:
        loader = DataLoader(file_path)
        df = loader.load()
        loader.validate_columns()
        summary = loader.get_summary()
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Rows: {summary['rows']}")
        print(f"   Columns: {', '.join(summary['columns'])}")
        print(f"\n📊 Category distribution:")
        for cat, count in summary['category_counts'].items():
            print(f"   - {cat}: {count}")
        print(f"\n💰 Amount statistics:")
        print(f"   - Min: {summary['amount_stats']['min']:.2f}")
        print(f"   - Max: {summary['amount_stats']['max']:.2f}")
        print(f"   - Mean: {summary['amount_stats']['mean']:.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
