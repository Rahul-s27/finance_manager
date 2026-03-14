"""
Save Best Model Module for Finance AutoML Manager.

Selects the best performing model from evaluation and saves it for later use.
"""

import os
import joblib


def save_best_model(results, models, vectorizer, model_dir="models"):
    """
    Select the best model and save it along with the vectorizer.
    
    Args:
        results: Dictionary with model names and accuracy scores
        models: Dictionary with trained models
        vectorizer: TF-IDF vectorizer used for feature extraction
        model_dir: Directory where models will be saved (default: "models")
        
    Returns:
        Dictionary with saved model paths
    """
    # Select best model (highest accuracy)
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_accuracy = results[best_model_name]
    
    print("\n" + "=" * 50)
    print("🏆 SELECTING BEST MODEL")
    print("=" * 50)
    print(f"   Best Model: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    # Create models folder if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"\n📁 Created folder: {model_dir}")
    
    # Save the best model
    model_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"   ✅ Saved model to: {model_path}")
    
    # Save the TF-IDF vectorizer
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"   ✅ Saved vectorizer to: {vectorizer_path}")
    
    # Save model info
    model_info = {
        "model_name": best_model_name,
        "accuracy": best_accuracy,
        "model_path": model_path,
        "vectorizer_path": vectorizer_path
    }
    
    print("\n" + "=" * 50)
    print("✅ MODEL SAVED SUCCESSFULLY")
    print("=" * 50)
    
    return model_info


if __name__ == "__main__":
    """
    Complete AutoML pipeline demo:
    1. Load data
    2. Clean data
    3. Create TF-IDF features
    4. Train multiple models
    5. Evaluate models
    6. Select and save best model
    """
    import sys
    import pandas as pd
    
    # Add pipeline to path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline'))
    
    from data_loader import DataLoader
    from preprocessing import DataCleaner
    from feature_engineering import create_features
    from train_models import train_models
    from evaluate_models import evaluate_models
    
    try:
        print("=" * 50)
        print("🤖 COMPLETE AUTOML PIPELINE")
        print("=" * 50)
        
        # Step 1: Load data
        print("\n📥 Step 1: Loading data...")
        df = pd.read_csv("data/training_data.csv")
        print(f"   ✅ Loaded {len(df)} rows")
        
        # Step 2: Clean data
        print("\n🧹 Step 2: Cleaning data...")
        cleaner = DataCleaner(df)
        clean_df = cleaner.clean()
        print(f"   ✅ Cleaned data: {len(clean_df)} rows remaining")
        
        # Step 3: Create TF-IDF features
        print("\n🔤 Step 3: Creating TF-IDF features...")
        X, y, vectorizer = create_features(clean_df, max_features=100)
        print(f"   ✅ Created {X.shape[1]} TF-IDF features")
        
        # Step 4: Train multiple models
        print("\n🏋️  Step 4: Training models...")
        models, X_test, y_test = train_models(X, y)
        
        # Step 5: Evaluate models
        print("\n📊 Step 5: Evaluating models...")
        results = evaluate_models(models, X_test, y_test)
        
        # Step 6: Select and save best model
        print("\n💾 Step 6: Saving best model...")
        model_info = save_best_model(results, models, vectorizer)
        
        print("\n" + "=" * 50)
        print("🎉 AUTOML PIPELINE COMPLETED!")
        print("=" * 50)
        print("\n📦 Saved files:")
        print(f"   - Model: {model_info['model_path']}")
        print(f"   - Vectorizer: {model_info['vectorizer_path']}")
        print(f"   - Best Model: {model_info['model_name']}")
        print(f"   - Accuracy: {model_info['accuracy']:.4f}")
        
        # Verify files were created
        print("\n🔍 Verifying saved files:")
        if os.path.exists(model_info['model_path']):
            size = os.path.getsize(model_info['model_path'])
            print(f"   ✅ {model_info['model_path']} ({size} bytes)")
        if os.path.exists(model_info['vectorizer_path']):
            size = os.path.getsize(model_info['vectorizer_path'])
            print(f"   ✅ {model_info['vectorizer_path']} ({size} bytes)")
        
        print("\n✅ Model is ready for predictions!")
        print("   Use joblib.load() to load the model in the backend.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
