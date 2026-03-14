"""
Test script for model evaluation step.

This script demonstrates the complete pipeline:
1. Load data
2. Clean data
3. Create TF-IDF features
4. Train multiple models
5. Evaluate and compare models
"""

import pandas as pd
import sys
import os

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline'))

from data_loader import DataLoader
from preprocessing import DataCleaner
from feature_engineering import create_features
from train_models import train_models
from evaluate_models import evaluate_models


# Load dataset
df = pd.read_csv("data/training_data.csv")
print(f"✅ Loaded {len(df)} rows")

# Clean data
cleaner = DataCleaner(df)
clean_df = cleaner.clean()
print(f"✅ Cleaned data: {len(clean_df)} rows remaining")

# Create TF-IDF features
X, y, vectorizer = create_features(clean_df, max_features=100)
print(f"✅ Created {X.shape[1]} TF-IDF features")

# Train multiple models
models, X_test, y_test = train_models(X, y)

# Evaluate models
print("\n" + "="*50)
print("📊 EVALUATION RESULTS")
print("="*50)

results = evaluate_models(models, X_test, y_test)

print("\n" + "="*50)
print("📊 SUMMARY")
print("="*50)
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n✅ Evaluation complete!")
