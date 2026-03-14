# Finance AutoML Manager

An automated machine learning system for classifying financial transactions into categories (food, transport, shopping, bills, fuel, subscription, groceries, and other expenses).

## Features

- **Data Upload**: Upload CSV files directly in the Streamlit UI
- **Auto-Cleaning**: Automatically cleans and standardizes transaction data
- **Feature Engineering**: Extracts text and numeric features automatically
- **Model Training**: Trains classification models with cross-validation
- **Model Saving**: Persists trained models for future predictions
- **Streamlit Frontend**: Interactive UI for predictions and visualizations

## Project Structure

```
finance-automl-manager/
├── data/
│   └── training_data.csv   # Training dataset
├── models/
│   ├── best_model.pkl
│   └── vectorizer.pkl
├── pipeline/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── evaluate_models.py
├── frontend/
│   └── streamlit_app.py
├── train_pipeline.py
├── test_dataset.py         # Quick dataset validation
└── requirements.txt        # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model

Train once and persist the best model + vectorizer:

```bash
python train_pipeline.py
```

This will:
- Load data from `data/training_data.csv`
- Clean and preprocess the data
- Create TF-IDF features
- Train multiple models and select the best
- Save:
  - `models/best_model.pkl`
  - `models/vectorizer.pkl`

### 3. Use Different Options

```bash
# Use custom dataset
python main.py --data path/to/your/data.csv

# Train with different model
python main.py --model logistic_regression

# Enable TF-IDF features
python main.py --tfidf --features 200

# Use larger test set
python main.py --test-size 0.3
```

### 3. Run the Streamlit App

```bash
streamlit run frontend/streamlit_app.py
```

## Data Format

The system expects CSV files with these columns:

| Column      | Description                      | Example              |
|-------------|----------------------------------|----------------------|
| description | Transaction text description     | "uber ride to office"|
| amount      | Transaction value (numeric)      | 250                  |
| category    | Category label for training      | transport            |

### Supported Categories

- **food** - Restaurants, cafes, food delivery
- **transport** - Uber, Ola, trains, flights, fuel
- **shopping** - Amazon, Flipkart, retail stores
- **bills** - Electricity, water, internet, phone
- **fuel** - Petrol, diesel, CNG
- **subscription** - Netflix, Prime, Spotify
- **groceries** - Bigbasket, supermarkets, vegetables
- **other_expenses** - Medical, gifts, entertainment

## Models Supported

- `random_forest` (default) - Good balance of speed and accuracy
- `logistic_regression` - Fast, interpretable
- `gradient_boosting` - High accuracy, slower
- `naive_bayes` - Fast baseline
- `svm` - Good for small datasets

## Development

### Testing Data Loading

```bash
python test_dataset.py
```

### Running Individual Pipeline Steps

```python
from pipeline.data_loader import DataLoader
from pipeline.preprocessing import DataCleaner
from pipeline.feature_engineering import FeatureEngineer
from pipeline.model_trainer import ModelTrainer

# Load
loader = DataLoader("data/training_data.csv")
df = loader.load()

# Clean
cleaner = DataCleaner(df)
clean_df = cleaner.clean()

# Engineer features
engineer = FeatureEngineer(clean_df)
X, y = engineer.prepare_for_training()

# Train
trainer = ModelTrainer()
results = trainer.run_full_pipeline(X, y)
```

## License

MIT License
