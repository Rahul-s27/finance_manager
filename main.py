"""
Finance AutoML Manager - Main Orchestrator

This script orchestrates the complete machine learning pipeline:
1. Data Loading
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Model Training
5. Model Saving

Usage:
    python main.py --data data/training_data.csv --model random_forest
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))

from data_loader import DataLoader, load_dataset
from preprocessing import DataCleaner, clean_data
from feature_engineering import FeatureEngineer, engineer_features
from model_trainer import ModelTrainer, load_model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Finance AutoML Manager - Train transaction classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                              # Use default training data
    python main.py --data custom_data.csv       # Use custom dataset
    python main.py --model logistic_regression  # Use specific model
    python main.py --tfidf --features 200       # Enable TF-IDF with 200 features
        """
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="data/training_data.csv",
        help="Path to training dataset (default: data/training_data.csv)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression", "gradient_boosting", 
                 "naive_bayes", "svm"],
        help="Model type to train (default: random_forest)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models",
        help="Directory to save trained models (default: models)"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)"
    )
    
    parser.add_argument(
        "--tfidf",
        action="store_true",
        help="Enable TF-IDF feature extraction"
    )
    
    parser.add_argument(
        "--features",
        type=int,
        default=100,
        help="Number of TF-IDF features (default: 100)"
    )
    
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip data cleaning (if data is already clean)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def print_banner():
    """Print application banner."""
    print("""
╔══════════════════════════════════════════════════════════╗
║           FINANCE AUTOML MANAGER                         ║
║       Automated Transaction Classification System        ║
╚══════════════════════════════════════════════════════════╝
    """)


def run_pipeline(args) -> Dict[str, Any]:
    """
    Execute the complete ML pipeline.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary with pipeline results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "data_path": args.data,
        "model_type": args.model,
        "success": False
    }
    
    try:
        # Step 1: Load Data
        print("\n" + "="*60)
        print("STEP 1: DATA LOADING")
        print("="*60)
        
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"Dataset not found: {args.data}")
        
        print(f"📁 Loading data from: {args.data}")
        loader = DataLoader(args.data)
        raw_df = loader.load()
        loader.validate_columns()
        
        summary = loader.get_summary()
        print(f"✅ Loaded {summary['rows']} rows with {len(summary['columns'])} columns")
        print(f"   Categories: {', '.join(summary['category_counts'].keys())}")
        
        results["data_loaded"] = {
            "rows": summary["rows"],
            "columns": summary["columns"],
            "category_distribution": summary["category_counts"]
        }
        
        # Step 2: Clean Data
        print("\n" + "="*60)
        print("STEP 2: DATA CLEANING")
        print("="*60)
        
        if args.skip_cleaning:
            print("⏭️  Skipping data cleaning (--skip-cleaning flag)")
            clean_df = raw_df
        else:
            cleaner = DataCleaner(raw_df)
            clean_df = cleaner.clean()
            cleaning_report = cleaner.get_cleaning_report()
            
            results["cleaning"] = {
                "original_rows": cleaning_report["original_rows"],
                "cleaned_rows": cleaning_report["cleaned_rows"],
                "removed_rows": cleaning_report["removed_rows"]
            }
        
        # Step 3: Feature Engineering
        print("\n" + "="*60)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*60)
        
        engineer = FeatureEngineer(clean_df)
        features_df = engineer.engineer_features(
            use_tfidf=args.tfidf,
            tfidf_max_features=args.features
        )
        
        X, y = engineer.prepare_for_training()
        
        results["features"] = {
            "total_features": X.shape[1],
            "samples": X.shape[0],
            "tfidf_enabled": args.tfidf,
            "tfidf_features": args.features if args.tfidf else 0
        }
        
        # Step 4: Model Training
        print("\n" + "="*60)
        print("STEP 4: MODEL TRAINING")
        print("="*60)
        
        trainer = ModelTrainer(
            model_dir=args.output,
            model_name=f"transaction_classifier_{args.model}"
        )
        
        training_results = trainer.run_full_pipeline(
            X=X,
            y=y,
            model_type=args.model,
            test_size=args.test_size
        )
        
        results["training"] = {
            "model_path": training_results["model_path"],
            "metadata_path": training_results["metadata_path"],
            "accuracy": training_results["metrics"]["accuracy"],
            "f1_score": training_results["metrics"]["f1_score"],
            "precision": training_results["metrics"]["precision"],
            "recall": training_results["metrics"]["recall"]
        }
        
        # Step 5: Summary
        print("\n" + "="*60)
        print("STEP 5: PIPELINE SUMMARY")
        print("="*60)
        
        print(f"""
🏆 Pipeline completed successfully!

📊 Dataset:
   - Total samples: {results['features']['samples']}
   - Features: {results['features']['total_features']}
   - Categories: {len(summary['category_counts'])}

🎯 Model Performance:
   - Accuracy:  {results['training']['accuracy']:.4f}
   - F1 Score:  {results['training']['f1_score']:.4f}
   - Precision: {results['training']['precision']:.4f}
   - Recall:    {results['training']['recall']:.4f}

💾 Saved Files:
   - Model:    {results['training']['model_path']}
   - Metadata: {results['training']['metadata_path']}
        """)
        
        results["success"] = True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        results["error"] = str(e)
        import traceback
        if args.verbose:
            traceback.print_exc()
        raise
    
    return results


def save_run_report(results: Dict[str, Any], output_dir: str = "models"):
    """Save pipeline run report to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"run_report_{timestamp}.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Run report saved: {report_path}")
    
    return report_path


def main():
    """Main entry point."""
    args = parse_arguments()
    
    print_banner()
    
    print(f"Configuration:")
    print(f"   Data path:    {args.data}")
    print(f"   Model type:   {args.model}")
    print(f"   Output dir:   {args.output}")
    print(f"   Test size:    {args.test_size}")
    print(f"   TF-IDF:       {'Enabled' if args.tfidf else 'Disabled'}")
    
    try:
        # Run pipeline
        results = run_pipeline(args)
        
        # Save report
        report_path = save_run_report(results, args.output)
        
        print(f"\n{'='*60}")
        print("🎉 All done! Model is ready for predictions.")
        print(f"{'='*60}\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
