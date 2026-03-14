"""
Model Training Module for Finance AutoML Manager.

Trains multiple machine learning algorithms and returns trained models for evaluation.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def train_models(X, y):
    """
    Train multiple machine learning models on the dataset.
    
    The dataset is split into:
    - 80% training data (used to train models)
    - 20% testing data (used to evaluate models)
    
    Args:
        X: Feature matrix (TF-IDF or other features)
        y: Target labels (categories)
        
    Returns:
        Tuple of (models, X_test, y_test) where:
        - models: Dictionary of trained models
        - X_test: Test feature matrix
        - y_test: Test labels
    """
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    
    models = {}
    
    # Train Logistic Regression
    print("\n   Training Logistic Regression...")
    models["logistic_regression"] = LogisticRegression(max_iter=1000)
    models["logistic_regression"].fit(X_train, y_train)
    print("   ✅ Logistic Regression trained")
    
    # Train Naive Bayes
    print("\n   Training Naive Bayes...")
    models["naive_bayes"] = MultinomialNB()
    models["naive_bayes"].fit(X_train, y_train)
    print("   ✅ Naive Bayes trained")
    
    # Train Random Forest
    print("\n   Training Random Forest...")
    models["random_forest"] = RandomForestClassifier(n_estimators=100, random_state=42)
    models["random_forest"].fit(X_train, y_train)
    print("   ✅ Random Forest trained")
    
    # Train Decision Tree
    print("\n   Training Decision Tree...")
    models["decision_tree"] = DecisionTreeClassifier(random_state=42)
    models["decision_tree"].fit(X_train, y_train)
    print("   ✅ Decision Tree trained")
    
    # Train SVM
    print("\n   Training SVM...")
    models["svm"] = SVC(random_state=42)
    models["svm"].fit(X_train, y_train)
    print("   ✅ SVM trained")
    
    print("\n" + "="*50)
    print("✅ All models trained successfully!")
    print(f"   Trained {len(models)} models: {list(models.keys())}")
    
    return models, X_test, y_test


if __name__ == "__main__":
    # Example usage with the dataset
    import sys
    import os
    import pandas as pd
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from pipeline.data_loader import DataLoader
    from pipeline.preprocessing import DataCleaner
    from pipeline.feature_engineering import create_features
    
    try:
        print("="*50)
        print("🤖 AutoML Model Training Demo")
        print("="*50)
        
        # Load data
        print("\n📥 Loading data...")
        file_path = os.path.join("data", "training_data.csv")
        loader = DataLoader(file_path)
        raw_df = loader.load()
        
        # Clean data
        print("\n🧹 Cleaning data...")
        cleaner = DataCleaner(raw_df)
        clean_df = cleaner.clean()
        
        # Create TF-IDF features
        print("\n🔤 Creating TF-IDF features...")
        X, y, vectorizer = create_features(clean_df, max_features=100)
        
        print(f"\n📊 Dataset info:")
        print(f"   Total samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Categories: {len(y.unique())}")
        
        # Train models
        print("\n🏋️  Training multiple models...")
        print("="*50)
        models, X_test, y_test = train_models(X, y)
        
        print("\n📦 Models ready for evaluation!")
        print(f"   Models dictionary keys: {list(models.keys())}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
