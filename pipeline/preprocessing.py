"""
Preprocessing Module for Finance AutoML Manager.

Handles data cleaning and preprocessing for transaction datasets.
"""

import pandas as pd
import re
import numpy as np
from typing import Optional, List, Dict


class DataCleaner:
    """Clean and preprocess financial transaction data."""
    
    # Column name variations for different banks
    COLUMN_MAPPINGS = {
        'description': ['description', 'narration', 'particulars', 'details', 'transaction details', 
                       'remarks', 'description of transaction', 'txn description', 'particular'],
        'amount': ['amount', 'debit', 'credit', 'transaction amount', 'txn amount', 'value'],
        'category': ['category', 'type', 'classification'],
        'date': ['date', 'transaction date', 'txn date', 'value date'],
        'merchant': ['merchant', 'merchant name', 'payee', 'recipient'],
        'transaction_type': ['type', 'transaction type', 'txn type', 'debit/credit', 'dr/cr']
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner.
        
        Args:
            df: Raw DataFrame to clean
        """
        self.raw_df = df.copy()
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.column_mapping: Dict[str, str] = {}
        
    def _detect_columns(self) -> Dict[str, str]:
        """
        Auto-detect columns based on common bank statement formats.
        
        Returns:
            Dictionary mapping standard column names to actual column names
        """
        df_columns_lower = {col.lower().strip(): col for col in self.raw_df.columns}
        detected = {}
        
        for standard_name, variations in self.COLUMN_MAPPINGS.items():
            for variation in variations:
                if variation in df_columns_lower:
                    detected[standard_name] = df_columns_lower[variation]
                    break
        
        return detected
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to standard names for consistency.
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            DataFrame with standardized column names
        """
        self.column_mapping = self._detect_columns()
        
        # Create reverse mapping (actual -> standard)
        rename_map = {actual: standard for standard, actual in self.column_mapping.items()}
        
        # Rename columns
        df = df.rename(columns=rename_map)
        
        print(f"   Detected columns: {list(self.column_mapping.keys())}")
        return df
    
    def _extract_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract amount from various formats (debit/credit columns, signed values).
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with standardized amount column
        """
        if 'amount' in df.columns:
            # Already have amount column
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        elif 'debit' in df.columns or 'credit' in df.columns:
            # Handle separate debit/credit columns
            df['debit'] = pd.to_numeric(df.get('debit', 0), errors='coerce').fillna(0)
            df['credit'] = pd.to_numeric(df.get('credit', 0), errors='coerce').fillna(0)
            
            # Create signed amount (negative for debit/expense, positive for credit/income)
            df['amount'] = df['credit'] - df['debit']
            
            # Add transaction type indicator
            df['transaction_type'] = df.apply(
                lambda x: 'expense' if x['debit'] > 0 else 'income' if x['credit'] > 0 else 'unknown',
                axis=1
            )
        
        return df
    
    def _extract_merchant_from_description(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract merchant/payee name from transaction description.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with extracted merchant column
        """
        if 'description' not in df.columns:
            return df
        
        # Common patterns in transaction descriptions
        # Pattern 1: UPI/IMPS transfers - extract after "/" or before numbers
        # Pattern 2: POS/Online purchases - extract merchant name
        # Pattern 3: NEFT/RTGS - extract beneficiary name
        
        def extract_merchant(desc):
            if pd.isna(desc):
                return "unknown"
            
            desc = str(desc).upper()
            
            # UPI pattern: UPI/merchant@upi or UPI/1234567890/merchant
            upi_match = re.search(r'UPI[/\-]([^/\s]+)', desc)
            if upi_match:
                merchant = upi_match.group(1)
                # Clean up @upi suffix
                merchant = re.sub(r'@.*$', '', merchant)
                return merchant.lower()
            
            # POS pattern: POS MERCHANT_NAME or POS-DEBIT MERCHANT
            pos_match = re.search(r'POS[-\s]*(DEBIT)?[-\s]*([A-Z][A-Z\s]+)', desc)
            if pos_match:
                return (pos_match.group(2) or pos_match.group(1) or "pos_merchant").lower().strip()
            
            # IMPS/NEFT pattern: IMPS-123456789012-Beneficiary Name
            imps_match = re.search(r'(?:IMPS|NEFT|RTGS)[-\s]*\d*[-\s]*([A-Z][A-Z\s]+)', desc)
            if imps_match:
                return imps_match.group(1).lower().strip()
            
            # Extract first meaningful word (often merchant name)
            # Skip common prefixes
            prefixes = ['UPI', 'POS', 'IMPS', 'NEFT', 'RTGS', 'NACH', 'ECS', 'ATM', 'CASH']
            words = desc.split()
            for word in words:
                clean_word = re.sub(r'[^A-Z]', '', word)
                if clean_word and clean_word not in prefixes and len(clean_word) > 2:
                    return clean_word.lower()
            
            return "unknown"
        
        df['merchant'] = df['description'].apply(extract_merchant)
        return df
    
    def _extract_transaction_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract transaction mode/type from description (UPI, POS, IMPS, etc.).
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with transaction mode column
        """
        if 'description' not in df.columns:
            return df
        
        def get_transaction_mode(desc):
            if pd.isna(desc):
                return "unknown"
            
            desc = str(desc).upper()
            
            modes = {
                'upi': 'UPI',
                'pos': 'POS',
                'imps': 'IMPS',
                'neft': 'NEFT',
                'rtgs': 'RTGS',
                'nach': 'NACH',
                'ecs': 'ECS',
                'atm': 'ATM',
                'cash': 'CASH'
            }
            
            for key, mode in modes.items():
                if key in desc:
                    return mode
            
            return "other"
        
        df['transaction_mode'] = df['description'].apply(get_transaction_mode)
        return df
    
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
        Standardize text by converting to lowercase and cleaning special characters.
        Preserves numbers and important identifiers.
        
        Args:
            df: DataFrame to process
            column: Column name containing text to clean
            
        Returns:
            DataFrame with standardized text
        """
        if column not in df.columns:
            print(f"   Warning: Column '{column}' not found, skipping text standardization")
            return df
        
        # Convert to lowercase
        df[column] = df[column].str.lower()
        
        # Remove excessive whitespace but keep alphanumeric characters
        # Numbers are important (merchant IDs, amounts, reference numbers)
        df[column] = df[column].str.strip()
        df[column] = df[column].str.replace(r'\s+', ' ', regex=True)
        
        # Remove only special characters that don't add meaning
        # Keep alphanumeric, spaces, and common separators
        df[column] = df[column].apply(
            lambda x: re.sub(r'[^\w\s/-]', ' ', str(x)) if pd.notna(x) else x
        )
        
        # Clean up multiple spaces
        df[column] = df[column].str.replace(r'\s+', ' ', regex=True).str.strip()
        
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
    
    # Category keywords for label cleaning
    CATEGORY_KEYWORDS = {
        'food': ['food', 'swiggy', 'zomato', 'restaurant', 'cafe', 'coffee', 'lunch', 'dinner', 
                'pizza', 'burger', 'meal', 'snack', 'dominos', 'starbucks', 'mcdonalds', 'kfc'],
        'transport': ['uber', 'ola', 'taxi', 'cab', 'bus', 'train', 'metro', 'ride', 'ticket',
                     'travel', 'trip', 'irctc', 'redbus'],
        'shopping': ['amazon', 'flipkart', 'shop', 'purchase', 'store', 'mart', 'clothes',
                    'shoes', 'electronics', 'fashion', 'mall', 'bookmyshow', 'pvr'],
        'grocery': ['grocery', 'supermarket', 'vegetable', 'fruit', 'milk', 'bread', 'bigbasket',
                   'grofers', 'dmart', 'reliance smart', 'blinkit', 'zepto'],
        'bills': ['bill', 'electricity', 'water', 'gas', 'internet', 'phone', 'recharge',
                 'payment', 'bescom', 'airtel', 'jio', 'msedcl'],
        'subscription': ['netflix', 'prime', 'spotify', 'subscription', 'membership',
                        'disney', 'hotstar', 'streaming', 'premium'],
        'fuel': ['petrol', 'diesel', 'cng', 'fuel', 'gas', 'pump', 'indianoil', 'bpcl', 'hpcl', 'shell'],
        'entertainment': ['movie', 'concert', 'gaming', 'steam', 'playstation', 'xbox', 'insider'],
    }
    
    def clean_labels(self, df: pd.DataFrame, description_col: str = 'description', 
                     category_col: str = 'category', threshold: float = 0.8) -> pd.DataFrame:
        """
        Detect and fix mislabeled transactions based on keyword matching.
        
        This helps when dataset has noisy/incoherent labels where descriptions
        don't match the assigned categories.
        
        Args:
            df: DataFrame with description and category columns
            description_col: Column containing transaction descriptions
            category_col: Column containing category labels
            threshold: Confidence threshold for relabeling (0-1)
            
        Returns:
            DataFrame with cleaned labels
        """
        if description_col not in df.columns or category_col not in df.columns:
            return df
        
        print("🔍 Checking for label noise...")
        
        corrected = 0
        df = df.copy()
        
        for idx, row in df.iterrows():
            desc = str(row[description_col]).lower()
            current_label = row[category_col]
            
            # Find best matching category based on keywords
            best_category = None
            best_score = 0
            
            for category, keywords in self.CATEGORY_KEYWORDS.items():
                score = sum(1 for kw in keywords if kw in desc)
                if score > best_score:
                    best_score = score
                    best_category = category
            
            # If strong keyword match contradicts current label, fix it
            if best_category and best_score >= 1:
                if current_label.lower() != best_category:
                    # Check if it's a clear mismatch (strong signal)
                    if best_score >= 2 or (best_score == 1 and len(desc.split()) <= 3):
                        df.at[idx, category_col] = best_category
                        corrected += 1
        
        if corrected > 0:
            print(f"   Fixed {corrected} mislabeled transactions ({corrected/len(df)*100:.1f}%)")
        else:
            print("   No obvious label errors detected")
        
        return df
    
    def clean(self, 
              text_column: str = "description",
              remove_missing: bool = True,
              standardize: bool = True,
              remove_dups: bool = True,
              clean_labels: bool = True) -> pd.DataFrame:
        """
        Run full cleaning pipeline on the dataset.
        
        Args:
            text_column: Column containing text descriptions (auto-detected if not found)
            remove_missing: Whether to remove rows with missing values
            standardize: Whether to standardize text
            remove_dups: Whether to remove duplicate rows
            
        Returns:
            Cleaned DataFrame
        """
        print("🧹 Starting data cleaning...")
        
        df = self.raw_df.copy()
        initial_rows = len(df)
        
        # Step 1: Detect and standardize column names
        df = self._standardize_columns(df)
        
        # Step 2: Extract amount from various formats
        df = self._extract_amount(df)
        
        # Step 3: Extract merchant names and transaction modes
        df = self._extract_merchant_from_description(df)
        df = self._extract_transaction_mode(df)
        
        # Step 4: Remove missing values
        if remove_missing:
            # Require at least description or merchant
            required_cols = [c for c in ['description', 'merchant'] if c in df.columns]
            if required_cols:
                df = df.dropna(subset=required_cols)
            df = df.dropna(subset=['amount']) if 'amount' in df.columns else df
        
        # Step 5: Standardize text
        if standardize and 'description' in df.columns:
            df = self.standardize_text(df, column="description")
        
        # Step 6: Clean labels if requested
        if clean_labels and 'category' in df.columns:
            df = self.clean_labels(df)
        
        # Step 7: Remove duplicates
        if remove_dups:
            dup_cols = [c for c in ['description', 'amount', 'date'] if c in df.columns]
            if len(dup_cols) >= 2:
                df = self.remove_duplicates(df, subset=dup_cols)
        
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
            "column_mapping": self.column_mapping,
            "missing_values": self.cleaned_df.isnull().sum().to_dict()
        }


def clean_data(df: pd.DataFrame, text_column: str = "description", clean_labels: bool = True) -> pd.DataFrame:
    """
    Convenience function to clean a DataFrame with smart column detection.
    
    Args:
        df: Raw DataFrame to clean
        text_column: Column containing text descriptions (auto-detected if not found)
        clean_labels: Whether to detect and fix mislabeled transactions
        
    Returns:
        Cleaned DataFrame with standardized columns
    """
    cleaner = DataCleaner(df)
    return cleaner.clean(text_column=text_column, clean_labels=clean_labels)


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
