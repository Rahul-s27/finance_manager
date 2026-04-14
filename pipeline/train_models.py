"""
Model Training Module for Finance AutoML Manager.

Trains multiple machine learning algorithms and returns trained models for evaluation.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def train_models(X, y):
    """
    Train multiple machine learning models on the dataset.
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        Tuple of (models, X_test, y_test) where:
        - models: Dictionary of trained models
        - X_test: Test feature matrix  
        - y_test: Test labels
    """
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    
    models = {}
    
    # Train Logistic Regression
    print("\n   Training Logistic Regression...")
    models["logistic_regression"] = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    models["logistic_regression"].fit(X_train, y_train)
    print("   Logistic Regression trained")
    
    # Train Random Forest
    print("\n   Training Random Forest...")
    models["random_forest"] = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    models["random_forest"].fit(X_train, y_train)
    print("   Random Forest trained")
    
    # Train Decision Tree
    print("\n   Training Decision Tree...")
    models["decision_tree"] = DecisionTreeClassifier(
        max_depth=15, min_samples_split=10, min_samples_leaf=5,
        random_state=42, class_weight='balanced'
    )
    models["decision_tree"].fit(X_train, y_train)
    print("   Decision Tree trained")
    
    # Train SVM
    print("\n   Training SVM...")
    models["svm"] = SVC(random_state=42, probability=True, class_weight='balanced', C=0.5)
    models["svm"].fit(X_train, y_train)
    print("   SVM trained")
    
    # Train Ensemble
    print("\n   Training Ensemble (Voting Classifier)...")
    models["ensemble"] = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
        ],
        voting='soft'
    )
    models["ensemble"].fit(X_train, y_train)
    print("   Ensemble trained")
    
    print("\n" + "="*50)
    print("All models trained successfully!")
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
        print("AutoML Model Training Demo")
        print("="*50)
        
        # Load data
        print("\nLoading data...")
        file_path = os.path.join("data", "training_data.csv")
        loader = DataLoader(file_path)
        raw_df = loader.load()
        
        # Clean data
        print("\nCleaning data...")
        cleaner = DataCleaner(raw_df)
        clean_df = cleaner.clean()
        
        # Create TF-IDF features
        print("\nCreating TF-IDF features...")
        X, y, vectorizer = create_features(clean_df, max_features=100)
        
        print(f"\nDataset info:")
        print(f"   Total samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Categories: {len(y.unique())}")
        
        # Train models
        print("\nTraining multiple models...")
        print("="*50)
        models, X_test, y_test = train_models(X, y)
        
        print("\nModels ready for evaluation!")
        print(f"   Models dictionary keys: {list(models.keys())}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
