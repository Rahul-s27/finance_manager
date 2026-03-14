"""
Feature Engineering Module for Finance AutoML Manager.

Extracts features from transaction text and scales numeric values for ML training.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re


def create_features(df, text_column="description", max_features=100):
    """
    Create TF-IDF features from transaction descriptions.
    
    This is a simplified function that converts text descriptions
    into numerical TF-IDF features for machine learning.
    
    Args:
        df: DataFrame with transaction data
        text_column: Column containing text descriptions (default: "description")
        max_features: Maximum number of TF-IDF features to create (default: 100)
        
    Returns:
        Tuple of (X, y, vectorizer) where:
        - X: TF-IDF feature matrix
        - y: Target category labels
        - vectorizer: The fitted TfidfVectorizer
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 1)
    )
    
    # Convert text descriptions into numeric features
    X = vectorizer.fit_transform(df[text_column])
    
    # Target labels
    y = df["category"]
    
    # Print feature information
    feature_names = vectorizer.get_feature_names_out()
    print(f"   Created TF-IDF features: {X.shape[1]} features")
    print(f"   Vocabulary size: {len(feature_names)}")
    print(f"   Sample vocabulary: {list(feature_names)[:10]}...")
    
    return X, y, vectorizer


class FeatureEngineer:
    """Extract and engineer features from transaction data."""
    
    # Category-specific keywords for feature extraction
    CATEGORY_KEYWORDS = {
        "food": ["food", "restaurant", "cafe", "coffee", "lunch", "dinner", "breakfast", 
                 "pizza", "burger", "sandwich", "meal", "snack", "swiggy", "zomato",
                 "dominos", "starbucks", "mcdonalds", "kfc", "subway", "taco", "bakery"],
        "transport": ["uber", "ola", "taxi", "cab", "bus", "train", "flight", "metro",
                      "transport", "ride", "travel", "journey", "trip", "ticket", "fuel",
                      "petrol", "diesel", "auto", "rickshaw"],
        "shopping": ["amazon", "flipkart", "shop", "purchase", "buy", "store", "mart",
                     "clothes", "shoes", "electronics", "gadgets", "fashion", "mall",
                     "retail", "online", "order", "delivery"],
        "groceries": ["grocery", "supermarket", "vegetable", "fruit", "milk", "bread",
                      "rice", "wheat", "flour", "sugar", "oil", "dairy", "meat", "fish",
                      "bigbasket", "grofers", "dmart", "kirana"],
        "bills": ["bill", "electricity", "water", "gas", "internet", "phone", "mobile",
                  "recharge", "payment", "utility", "broadband", "wifi", "cable", "tv",
                  "subscription", "rent", "emi"],
        "subscription": ["netflix", "prime", "spotify", "subscription", "monthly", 
                         "annual", "membership", "disney", "hulu", "streaming", 
                         "premium", "plan"],
        "fuel": ["petrol", "diesel", "cng", "fuel", "gas", "station", "pump", 
                 "tank", "fill", "refill", "hp", "bp", "indian oil"],
        "other_expenses": ["medical", "doctor", "pharmacy", "medicine", "gym", "fitness",
                          "movie", "entertainment", "gift", "donation", "charity",
                          "insurance", "tax", "legal", "consulting"]
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FeatureEngineer.
        
        Args:
            df: Cleaned DataFrame with description, amount, and category columns
        """
        self.df = df.copy()
        self.features_df: Optional[pd.DataFrame] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.amount_scaler: Optional[StandardScaler] = None
    
    def extract_text_features(self, text_column: str = "description") -> pd.DataFrame:
        """
        Extract features from text descriptions.
        
        Features:
        - Word count
        - Character count
        - Average word length
        - Category keyword matches
        
        Args:
            text_column: Column containing text descriptions
            
        Returns:
            DataFrame with text features
        """
        df = self.df.copy()
        
        # Basic text statistics
        df["word_count"] = df[text_column].apply(lambda x: len(str(x).split()))
        df["char_count"] = df[text_column].apply(lambda x: len(str(x)))
        df["avg_word_length"] = df[text_column].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        
        # Category keyword features
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            df[f"has_{category}_keyword"] = df[text_column].apply(
                lambda x: any(kw in str(x).lower() for kw in keywords)
            ).astype(int)
            
            df[f"{category}_keyword_count"] = df[text_column].apply(
                lambda x: sum(str(x).lower().count(kw) for kw in keywords)
            )
        
        print(f"   Extracted {len([c for c in df.columns if c != text_column and c not in ['amount', 'category']])} text features")
        
        return df
    
    def extract_amount_features(self, amount_column: str = "amount") -> pd.DataFrame:
        """
        Extract features from amount values.
        
        Features:
        - Log amount (for skewed distribution)
        - Amount bins (small, medium, large)
        - Amount percentiles
        
        Args:
            amount_column: Column containing amount values
            
        Returns:
            DataFrame with amount features
        """
        df = self.df.copy()
        
        # Log transformation (handle negative and zero values)
        df["log_amount"] = df[amount_column].apply(
            lambda x: np.log1p(abs(x)) if pd.notna(x) else 0
        )
        
        # Amount categories based on distribution
        amount_median = df[amount_column].median()
        amount_q75 = df[amount_column].quantile(0.75)
        
        df["amount_category"] = df[amount_column].apply(
            lambda x: "small" if x <= amount_median 
            else "medium" if x <= amount_q75 
            else "large"
        )
        
        # One-hot encode amount categories
        amount_dummies = pd.get_dummies(df["amount_category"], prefix="amount")
        df = pd.concat([df, amount_dummies], axis=1)
        
        print(f"   Extracted amount features: log_amount, amount_category, amount bins")
        
        return df
    
    def create_tfidf_features(self, 
                              text_column: str = "description", 
                              max_features: int = 100,
                              ngram_range: tuple = (1, 2)) -> np.ndarray:
        """
        Create TF-IDF features from text descriptions.
        
        Args:
            text_column: Column containing text descriptions
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams to extract
            
        Returns:
            TF-IDF feature matrix
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            min_df=2,
            max_df=0.95
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df[text_column])
        
        print(f"   Created TF-IDF features: {tfidf_matrix.shape[1]} features")
        
        return tfidf_matrix
    
    def scale_amount(self, 
                     amount_column: str = "amount",
                     method: str = "standard") -> pd.DataFrame:
        """
        Scale amount values using StandardScaler or MinMaxScaler.
        
        Args:
            amount_column: Column containing amount values
            method: Scaling method ("standard" or "minmax")
            
        Returns:
            DataFrame with scaled amount
        """
        df = self.df.copy()
        
        if method == "standard":
            self.amount_scaler = StandardScaler()
        else:
            self.amount_scaler = MinMaxScaler()
        
        df["amount_scaled"] = self.amount_scaler.fit_transform(
            df[[amount_column]]
        )
        
        print(f"   Scaled amounts using {method} scaler")
        
        return df
    
    def engineer_features(self, 
                          text_column: str = "description",
                          amount_column: str = "amount",
                          use_tfidf: bool = False,
                          tfidf_max_features: int = 100) -> pd.DataFrame:
        """
        Run full feature engineering pipeline.
        
        Args:
            text_column: Column containing text descriptions
            amount_column: Column containing amount values
            use_tfidf: Whether to create TF-IDF features
            tfidf_max_features: Maximum TF-IDF features if enabled
            
        Returns:
            DataFrame with all engineered features
        """
        print("⚙️  Starting feature engineering...")
        
        # Start with original data and accumulate features
        df = self.df.copy()
        
        # Step 1: Extract text features
        text_features_df = self.extract_text_features(text_column)
        # Get only the new text feature columns (excluding original columns)
        text_feature_cols = [c for c in text_features_df.columns 
                            if c not in df.columns and c not in [text_column, amount_column, "category"]]
        for col in text_feature_cols:
            df[col] = text_features_df[col]
        
        # Step 2: Extract amount features
        amount_features_df = self.extract_amount_features(amount_column)
        # Get only the new amount feature columns
        amount_feature_cols = [c for c in amount_features_df.columns 
                              if c not in df.columns and c not in [text_column, amount_column, "category"]]
        for col in amount_feature_cols:
            df[col] = amount_features_df[col]
        
        # Step 3: Scale amount
        # Update self.df temporarily for scale_amount to work on current df
        original_df = self.df
        self.df = df
        scaled_df = self.scale_amount(amount_column, method="standard")
        self.df = original_df  # Restore
        # Get only the new scaled column
        if "amount_scaled" in scaled_df.columns and "amount_scaled" not in df.columns:
            df["amount_scaled"] = scaled_df["amount_scaled"]
        
        # Add TF-IDF features if requested
        if use_tfidf:
            tfidf_matrix = self.create_tfidf_features(
                text_column, 
                max_features=tfidf_max_features
            )
            # Convert to DataFrame and merge
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
                index=df.index
            )
            df = pd.concat([df, tfidf_df], axis=1)
        
        self.features_df = df
        
        feature_count = len([c for c in df.columns 
                            if c not in [text_column, amount_column, "category", "amount_category"]])
        
        print(f"✅ Feature engineering complete!")
        print(f"   Total features: {feature_count}")
        print(f"   Text features: word_count, char_count, keyword matches")
        print(f"   Numeric features: log_amount, scaled_amount, amount bins")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of engineered feature names (excluding original columns).
        
        Returns:
            List of feature column names
        """
        if self.features_df is None:
            raise RuntimeError("Features not engineered yet. Call engineer_features() first.")
        
        exclude = {"description", "amount", "category", "amount_category"}
        return [col for col in self.features_df.columns if col not in exclude]
    
    def prepare_for_training(self) -> tuple:
        """
        Prepare features and target for model training.
        
        Returns:
            Tuple of (X, y) where X is feature matrix and y is target labels
        """
        if self.features_df is None:
            raise RuntimeError("Features not engineered yet. Call engineer_features() first.")
        
        # Get feature columns (exclude original and target)
        feature_cols = self.get_feature_names()
        X = self.features_df[feature_cols]
        y = self.features_df["category"]
        
        print(f"   Ready for training: X shape = {X.shape}, y shape = {y.shape}")
        
        return X, y


def engineer_features(df: pd.DataFrame, 
                      use_tfidf: bool = False,
                      tfidf_max_features: int = 100) -> pd.DataFrame:
    """
    Convenience function to engineer features from a DataFrame.
    
    Args:
        df: Cleaned DataFrame with description, amount, and category
        use_tfidf: Whether to create TF-IDF features
        tfidf_max_features: Maximum TF-IDF features if enabled
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer(df)
    return engineer.engineer_features(use_tfidf=use_tfidf, 
                                      tfidf_max_features=tfidf_max_features)


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_loader import DataLoader
    from preprocessing import DataCleaner
    
    try:
        # Load and clean data
        file_path = os.path.join("..", "data", "training_data.csv")
        loader = DataLoader(file_path)
        raw_df = loader.load()
        
        cleaner = DataCleaner(raw_df)
        clean_df = cleaner.clean()
        
        print()
        
        # Engineer features
        engineer = FeatureEngineer(clean_df)
        features_df = engineer.engineer_features(use_tfidf=True, tfidf_max_features=50)
        
        # Prepare for training
        X, y = engineer.prepare_for_training()
        
        print()
        print("📊 Feature matrix sample:")
        print(X.head())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
