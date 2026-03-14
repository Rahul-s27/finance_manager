"""
Model Trainer Module for Finance AutoML Manager.

Trains classification models and saves them for future predictions.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder


class ModelTrainer:
    """Train and save machine learning models for transaction classification."""
    
    AVAILABLE_MODELS = {
        "random_forest": RandomForestClassifier,
        "logistic_regression": LogisticRegression,
        "gradient_boosting": GradientBoostingClassifier,
        "naive_bayes": MultinomialNB,
        "svm": SVC
    }
    
    DEFAULT_PARAMS = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1
        },
        "logistic_regression": {
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 3,
            "random_state": 42
        },
        "naive_bayes": {},
        "svm": {
            "kernel": "rbf",
            "random_state": 42,
            "probability": True
        }
    }
    
    def __init__(self, 
                 model_dir: str = "models",
                 model_name: str = "transaction_classifier"):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_dir: Directory to save trained models
            model_name: Base name for saved model files
        """
        self.model_dir = model_dir
        self.model_name = model_name
        self.model: Optional[Any] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.training_metadata: Dict[str, Any] = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_data(self, 
                     X: pd.DataFrame, 
                     y: pd.Series,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Classes: {len(self.label_encoder.classes_)}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, 
              X_train: pd.DataFrame, 
              y_train: np.ndarray,
              model_type: str = "random_forest",
              hyperparams: Optional[Dict] = None) -> Any:
        """
        Train a classification model.
        
        Args:
            X_train: Training features
            y_train: Training labels (encoded)
            model_type: Type of model to train
            hyperparams: Optional custom hyperparameters
            
        Returns:
            Trained model instance
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {', '.join(self.AVAILABLE_MODELS.keys())}"
            )
        
        # Get model class and parameters
        model_class = self.AVAILABLE_MODELS[model_type]
        params = hyperparams or self.DEFAULT_PARAMS.get(model_type, {})
        
        print(f"   Training {model_type} model...")
        
        # Initialize and train model
        self.model = model_class(**params)
        self.model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.model
    
    def evaluate(self, 
                 X_test: pd.DataFrame, 
                 y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels (encoded)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("No model trained yet. Call train() first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        
        # Decode labels for report
        y_test_decoded = self.label_encoder.inverse_transform(y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        
        print(f"\n📊 Model Performance:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "predictions": y_pred_decoded.tolist(),
            "true_labels": y_test_decoded.tolist()
        }
    
    def save_model(self, custom_name: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            custom_name: Optional custom name for the model file
            
        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise RuntimeError("No model trained yet. Call train() first.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = custom_name or self.model_name
        model_filename = f"{name}_{timestamp}.pkl"
        model_path = os.path.join(self.model_dir, model_filename)
        
        # Prepare model package
        model_package = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "metadata": self.training_metadata,
            "feature_names": getattr(self, "feature_names", None),
            "saved_at": timestamp
        }
        
        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model_package, f)
        
        print(f"   Model saved to: {model_path}")
        
        return model_path
    
    def save_metadata(self, filepath: str, metrics: Dict[str, Any]) -> str:
        """
        Save training metadata and metrics to JSON.
        
        Args:
            filepath: Path where model was saved
            metrics: Evaluation metrics dictionary
            
        Returns:
            Path to metadata file
        """
        metadata = {
            "model_path": filepath,
            "training_timestamp": datetime.now().isoformat(),
            "model_type": type(self.model).__name__ if self.model else None,
            "num_classes": len(self.label_encoder.classes_) if self.label_encoder else 0,
            "class_names": self.label_encoder.classes_.tolist() if self.label_encoder else [],
            "metrics": {
                k: v for k, v in metrics.items() 
                if k not in ["predictions", "true_labels"]
            },
            **self.training_metadata
        }
        
        # Save as JSON
        metadata_path = filepath.replace(".pkl", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Metadata saved to: {metadata_path}")
        
        return metadata_path
    
    def run_full_pipeline(self,
                         X: pd.DataFrame,
                         y: pd.Series,
                         model_type: str = "random_forest",
                         hyperparams: Optional[Dict] = None,
                         test_size: float = 0.2) -> Dict[str, Any]:
        """
        Run complete training pipeline: prepare, train, evaluate, save.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to train
            hyperparams: Optional custom hyperparameters
            test_size: Proportion for test split
            
        Returns:
            Dictionary with training results and file paths
        """
        print("\n🏋️  Starting Model Training Pipeline")
        print("=" * 50)
        
        # Prepare data
        print("\n📥 Preparing data...")
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, test_size=test_size)
        
        # Store feature names for prediction
        self.feature_names = list(X.columns)
        
        # Train model
        print("\n🎯 Training model...")
        self.train(X_train, y_train, model_type=model_type, hyperparams=hyperparams)
        
        # Evaluate
        print("\n📊 Evaluating model...")
        metrics = self.evaluate(X_test, y_test)
        
        # Save
        print("\n💾 Saving model...")
        model_path = self.save_model()
        metadata_path = self.save_metadata(model_path, metrics)
        
        print("\n" + "=" * 50)
        print("✅ Training pipeline completed successfully!")
        
        return {
            "model_path": model_path,
            "metadata_path": metadata_path,
            "metrics": metrics,
            "model_type": model_type,
            "num_features": X.shape[1],
            "num_samples": len(X)
        }


def load_model(model_path: str) -> Dict[str, Any]:
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to the saved model pickle file
        
    Returns:
        Dictionary containing model, label encoder, and metadata
    """
    with open(model_path, "rb") as f:
        model_package = pickle.load(f)
    
    return model_package


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_loader import DataLoader
    from preprocessing import DataCleaner
    from feature_engineering import FeatureEngineer
    
    try:
        # Load data
        file_path = os.path.join("..", "data", "training_data.csv")
        loader = DataLoader(file_path)
        raw_df = loader.load()
        
        # Clean data
        cleaner = DataCleaner(raw_df)
        clean_df = cleaner.clean()
        
        # Engineer features
        engineer = FeatureEngineer(clean_df)
        features_df = engineer.engineer_features(use_tfidf=False)
        X, y = engineer.prepare_for_training()
        
        # Train model
        trainer = ModelTrainer()
        results = trainer.run_full_pipeline(X, y, model_type="random_forest")
        
        print(f"\n📁 Model saved at: {results['model_path']}")
        print(f"📊 Final Accuracy: {results['metrics']['accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
