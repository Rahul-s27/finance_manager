"""
Model Evaluation Module for Finance AutoML Manager.

Compares multiple trained models and measures their classification performance.
"""

from sklearn.metrics import accuracy_score, classification_report


def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple trained models on the test dataset.
    
    For each model:
    - Predicts categories for test transactions
    - Compares predictions with actual labels
    - Calculates accuracy and detailed metrics
    
    Args:
        models: Dictionary of trained models {name: model}
        X_test: Test feature matrix
        y_test: Test labels (actual categories)
        
    Returns:
        Dictionary with model names as keys and accuracy scores as values
    """
    results = {}
    
    for name, model in models.items():
        # Make predictions on test data
        predictions = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        
        # Store result
        results[name] = accuracy
        
        # Print evaluation metrics
        print(f"\nModel: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, predictions, zero_division=0))
    
    return results


def print_model_comparison(results_dict):
    """
    Print a comparison table of all model performances.
    
    Args:
        results_dict: Dictionary with model names and accuracy scores
    """
    print("\n📈 Model Comparison Summary")
    print("=" * 50)
    print(f"{'Model':<25} {'Accuracy':>15}")
    print("-" * 50)
    
    # Sort by accuracy (descending)
    sorted_models = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
    
    for name, accuracy in sorted_models:
        print(f"{name:<25} {accuracy*100:>14.2f}%")
    
    # Show best model
    best_model = sorted_models[0][0]
    print("=" * 50)
    print(f"🏆 Best Model: {best_model}")


if __name__ == "__main__":
    # Example usage with the dataset
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from pipeline.data_loader import DataLoader
    from pipeline.preprocessing import DataCleaner
    from pipeline.feature_engineering import create_features
    from pipeline.train_models import train_models
    
    try:
        print("=" * 50)
        print("🤖 AutoML Model Evaluation Demo")
        print("=" * 50)
        
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
        print("\n🏋️  Training models...")
        models, X_test, y_test = train_models(X, y)
        
        # Evaluate models
        print("\n📊 Evaluating models...")
        results = evaluate_models(models, X_test, y_test)
        
        # Print comparison
        print_model_comparison(results)
        
        print("\n✅ Evaluation complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
