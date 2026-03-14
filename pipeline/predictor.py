"""
Predictor Module for Finance AutoML Manager.

Loads saved models and makes predictions on new transaction data.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional, Union

from data_loader import DataLoader
from preprocessing import DataCleaner
from feature_engineering import FeatureEngineer


class Predictor:
    """Load trained models and make predictions on new transactions."""
    
    def __init__(self, model_path: str):
        """
        Initialize the Predictor with a saved model.
        
        Args:
            model_path: Path to the saved model pickle file
        """
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.metadata = None
        self.feature_names = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the saved model package from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, "rb") as f:
            model_package = pickle.load(f)
        
        self.model = model_package["model"]
        self.label_encoder = model_package["label_encoder"]
        self.metadata = model_package.get("metadata", {})
        self.feature_names = model_package.get("feature_names", None)
        
        print(f"✅ Model loaded from: {self.model_path}")
        print(f"   Model type: {type(self.model).__name__}")
        print(f"   Classes: {list(self.label_encoder.classes_)}")
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from raw transaction data.
        
        Args:
            df: Raw DataFrame with description and amount columns
            
        Returns:
            DataFrame with engineered features only (no original columns)
        """
        # Clean data
        cleaner = DataCleaner(df)
        clean_df = cleaner.clean()
        
        # Engineer features
        engineer = FeatureEngineer(clean_df)
        features_df = engineer.engineer_features(use_tfidf=False)
        
        # Define columns to exclude (only original + target columns)
        exclude_cols = {"description", "amount", "category"}
        
        # Get feature columns only
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]
        
        # Ensure we have the same features as training
        if self.feature_names:
            # Add missing features with default values
            for feature in self.feature_names:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            
            # Select only the features used during training
            feature_cols = [f for f in self.feature_names if f not in exclude_cols]
        
        return features_df[feature_cols]
    
    def predict(self, 
                descriptions: Union[str, List[str]], 
                amounts: Union[float, List[float]]) -> Dict[str, Any]:
        """
        Make predictions on new transaction data.
        
        Args:
            descriptions: Transaction description(s) - single string or list
            amounts: Transaction amount(s) - single float or list
            
        Returns:
            Dictionary with predictions, probabilities, and confidence scores
        """
        # Convert single inputs to lists
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        if isinstance(amounts, (int, float)):
            amounts = [amounts]
        
        # Ensure equal length
        if len(descriptions) != len(amounts):
            raise ValueError("descriptions and amounts must have the same length")
        
        # Create DataFrame
        df = pd.DataFrame({
            "description": descriptions,
            "amount": amounts,
            "category": "unknown"  # Placeholder for required column
        })
        
        # Prepare features
        features_df = self._prepare_features(df)
        
        # Remove target column if present
        if "category" in features_df.columns:
            X = features_df.drop("category", axis=1)
        else:
            X = features_df
        
        # Make predictions
        predictions_encoded = self.model.predict(X)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Get prediction probabilities if available
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)
            confidence_scores = np.max(probabilities, axis=1)
            
            # Get top 3 predictions for each
            top_predictions = []
            for i, probs in enumerate(probabilities):
                top_indices = np.argsort(probs)[-3:][::-1]
                top_classes = self.label_encoder.inverse_transform(top_indices)
                top_scores = probs[top_indices]
                
                top_predictions.append([
                    {"category": cls, "probability": float(score)}
                    for cls, score in zip(top_classes, top_scores)
                ])
        else:
            confidence_scores = [None] * len(predictions)
            top_predictions = [[{"category": pred, "probability": None}] for pred in predictions]
        
        return {
            "predictions": predictions.tolist(),
            "confidence": confidence_scores.tolist() if isinstance(confidence_scores, np.ndarray) else confidence_scores,
            "top_3": top_predictions,
            "input_descriptions": descriptions,
            "input_amounts": amounts,
            "model_version": self.metadata.get("model_type", "unknown"),
            "saved_at": self.metadata.get("saved_at", "unknown")
        }
    
    def predict_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Make predictions on transactions from a CSV/Excel file.
        
        Args:
            file_path: Path to the transaction file
            
        Returns:
            DataFrame with original data and predicted categories
        """
        # Load data
        loader = DataLoader(file_path)
        df = loader.load()
        
        print(f"📁 Loaded {len(df)} transactions from {file_path}")
        
        # Make predictions
        result = self.predict(
            descriptions=df["description"].tolist(),
            amounts=df["amount"].tolist()
        )
        
        # Add predictions to original dataframe
        df["predicted_category"] = result["predictions"]
        df["confidence"] = result["confidence"]
        
        return df
    
    def batch_predict(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions on a batch of transactions.
        
        Args:
            transactions: List of dicts with 'description' and 'amount' keys
            
        Returns:
            List of prediction results
        """
        descriptions = [t["description"] for t in transactions]
        amounts = [t["amount"] for t in transactions]
        
        result = self.predict(descriptions, amounts)
        
        # Format results
        formatted_results = []
        for i in range(len(transactions)):
            formatted_results.append({
                "description": transactions[i]["description"],
                "amount": transactions[i]["amount"],
                "predicted_category": result["predictions"][i],
                "confidence": result["confidence"][i],
                "top_3": result["top_3"][i]
            })
        
        return formatted_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "model_type": type(self.model).__name__ if self.model else None,
            "classes": self.label_encoder.classes_.tolist() if self.label_encoder else [],
            "num_classes": len(self.label_encoder.classes_) if self.label_encoder else 0,
            "metadata": self.metadata
        }


def find_latest_model(model_dir: str = "models") -> Optional[str]:
    """
    Find the most recently saved model in a directory.
    
    Args:
        model_dir: Directory containing saved models
        
    Returns:
        Path to the latest model file, or None if not found
    """
    if not os.path.exists(model_dir):
        return None
    
    model_files = [
        f for f in os.listdir(model_dir) 
        if f.endswith(".pkl") and not f.endswith("_metadata.json")
    ]
    
    if not model_files:
        return None
    
    # Sort by modification time (most recent first)
    model_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(model_dir, f)),
        reverse=True
    )
    
    return os.path.join(model_dir, model_files[0])


if __name__ == "__main__":
    # CLI usage example
    import argparse
    
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument("--model", "-m", type=str, help="Path to saved model (auto-detect if not specified)")
    parser.add_argument("--file", "-f", type=str, help="CSV/Excel file with transactions to classify")
    parser.add_argument("--description", "-d", type=str, help="Single transaction description")
    parser.add_argument("--amount", "-a", type=float, help="Single transaction amount")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to search for models")
    parser.add_argument("--output", "-o", type=str, help="Output file for predictions (CSV)")
    
    args = parser.parse_args()
    
    try:
        # Find model
        if args.model:
            model_path = args.model
        else:
            model_path = find_latest_model(args.model_dir)
            if not model_path:
                print(f"❌ No model found in {args.model_dir}")
                print("   Train a model first with: python main.py")
                exit(1)
        
        # Initialize predictor
        predictor = Predictor(model_path)
        
        # Make predictions
        if args.file:
            # Batch prediction from file
            result_df = predictor.predict_from_file(args.file)
            
            print(f"\n📊 Predictions:")
            print(result_df[["description", "amount", "predicted_category", "confidence"]].to_string())
            
            # Save if output specified
            if args.output:
                result_df.to_csv(args.output, index=False)
                print(f"\n💾 Results saved to: {args.output}")
        
        elif args.description and args.amount is not None:
            # Single prediction
            result = predictor.predict(args.description, args.amount)
            
            print(f"\n🔮 Prediction Result:")
            print(f"   Description: {args.description}")
            print(f"   Amount: ₹{args.amount}")
            print(f"   Predicted Category: {result['predictions'][0]}")
            print(f"   Confidence: {result['confidence'][0]:.2%}" if result['confidence'][0] else "   Confidence: N/A")
            
            print(f"\n📋 Top 3 Categories:")
            for i, pred in enumerate(result['top_3'][0][:3], 1):
                prob_str = f"{pred['probability']:.2%}" if pred['probability'] else "N/A"
                print(f"   {i}. {pred['category']} ({prob_str})")
        
        else:
            print("❌ Please provide either --file or both --description and --amount")
            parser.print_help()
            exit(1)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
