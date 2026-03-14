"""
Preprocessing Module for Finance AutoML Manager.

Handles data cleaning and preprocessing for transaction datasets.
"""

import pandas as pd
import re
from typing import Optional


class DataCleaner:
    """Clean and preprocess financial transaction data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner.
        
        Args:
            df: Raw DataFrame to clean
        """
        self.raw_df = df.copy()
        self.cleaned_df: Optional[pd.DataFrame] = None
    
    def remove_missing_values(self, subset: Optional[list] = None) -> pd.DataFrame:
        """
        Remove rows with missing values.
        
        Args:
            subset: Optional list of columns to check for missing values
            
        Returns:
            DataFrame with missing values removed
        """
        before_count = len(self.raw_df)
        df = self.raw_df.dropna(subset=subset)
        after_count = len(df)
        removed = before_count - after_count
        
        if removed > 0:
            print(f"   Removed {removed} rows with missing values")
        
        return df
    
    def standardize_text(self, df: pd.DataFrame, column: str = "description") -> pd.DataFrame:
        """
        Standardize text by converting to lowercase and removing special characters.
        
        Args:
            df: DataFrame to process
            column: Column name containing text to clean
            
        Returns:
            DataFrame with standardized text
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        # Convert to lowercase
        df[column] = df[column].str.lower()
        
        # Remove special characters and numbers, keep only letters and spaces
        df[column] = df[column].apply(
            lambda x: re.sub(r'[^a-zA-Z ]', '', str(x)) if pd.notna(x) else x
        )
        
        # Remove extra whitespace
        df[column] = df[column].str.strip()
        df[column] = df[column].str.replace(r'\s+', ' ', regex=True)
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            df: DataFrame to process
            subset: Optional list of columns to consider for duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        before_count = len(df)
        df = df.drop_duplicates(subset=subset, keep="first")
        after_count = len(df)
        removed = before_count - after_count
        
        if removed > 0:
            print(f"   Removed {removed} duplicate rows")
        
        return df
    
    def clean(self, 
              text_column: str = "description",
              remove_missing: bool = True,
              standardize: bool = True,
              remove_dups: bool = True) -> pd.DataFrame:
        """
        Run full cleaning pipeline on the dataset.
        
        Args:
            text_column: Column containing text descriptions
            remove_missing: Whether to remove rows with missing values
            standardize: Whether to standardize text
            remove_dups: Whether to remove duplicate rows
            
        Returns:
            Cleaned DataFrame
        """
        print("🧹 Starting data cleaning...")
        
        df = self.raw_df.copy()
        initial_rows = len(df)
        
        if remove_missing:
            df = self.remove_missing_values()
        
        if standardize:
            df = self.standardize_text(df, column=text_column)
        
        if remove_dups:
            df = self.remove_duplicates(df)
        
        final_rows = len(df)
        removed = initial_rows - final_rows
        
        self.cleaned_df = df
        
        print(f"✅ Cleaning complete!")
        print(f"   Initial rows: {initial_rows}")
        print(f"   Final rows: {final_rows}")
        print(f"   Removed: {removed} rows ({removed/initial_rows*100:.1f}%)")
        
        return df
    
    def get_cleaning_report(self) -> dict:
        """
        Get a report of the cleaning process.
        
        Returns:
            Dictionary with cleaning statistics
        """
        if self.cleaned_df is None:
            raise RuntimeError("Data not cleaned yet. Call clean() first.")
        
        return {
            "original_rows": len(self.raw_df),
            "cleaned_rows": len(self.cleaned_df),
            "removed_rows": len(self.raw_df) - len(self.cleaned_df),
            "columns": list(self.cleaned_df.columns),
            "missing_values": self.cleaned_df.isnull().sum().to_dict()
        }


def clean_data(df: pd.DataFrame, text_column: str = "description") -> pd.DataFrame:
    """
    Convenience function to clean a DataFrame.
    
    Args:
        df: Raw DataFrame to clean
        text_column: Column containing text descriptions
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner(df)
    return cleaner.clean(text_column=text_column)


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_loader import DataLoader
    
    # Default training data path
    file_path = os.path.join("..", "data", "training_data.csv")
    
    try:
        # Load data
        loader = DataLoader(file_path)
        df = loader.load()
        
        print(f"📊 Loaded dataset with {len(df)} rows")
        print(f"   Columns: {', '.join(df.columns)}")
        print()
        
        # Clean data
        cleaner = DataCleaner(df)
        cleaned_df = cleaner.clean()
        
        print()
        print("🔍 Sample cleaned descriptions:")
        for i, desc in enumerate(cleaned_df["description"].head(5)):
            print(f"   {i+1}. {desc}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
