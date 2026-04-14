"""
Fix dataset labels to match descriptions.
This script corrects incoherent labels in the synthetic dataset.
"""

import pandas as pd
import sys
from pathlib import Path

# Category keywords for label matching
CATEGORY_KEYWORDS = {
    'food': ['food', 'swiggy', 'zomato', 'restaurant', 'cafe', 'coffee', 'lunch', 'dinner', 
            'pizza', 'burger', 'meal', 'snack', 'dominos', 'starbucks', 'mcdonalds', 'kfc'],
    'transport': ['uber', 'ola', 'taxi', 'cab', 'bus', 'train', 'metro', 'ride', 'ticket',
                 'travel', 'trip', 'irctc', 'redbus'],
    'shopping': ['amazon', 'flipkart', 'shop', 'purchase', 'store', 'mart', 'clothes',
                'shoes', 'electronics', 'fashion', 'mall', 'bookmyshow', 'pvr'],
    'grocery': ['grocery', 'supermarket', 'vegetable', 'fruit', 'milk', 'bread', 'bigbasket',
               'grofers', 'dmart', 'reliance smart', 'blinkit', 'zepto'],
    'bills': ['bill', 'electricity', 'water', 'gas', 'internet', 'phone', 'recharge',
             'payment', 'bescom', 'airtel', 'jio', 'msedcl'],
    'subscription': ['netflix', 'prime', 'spotify', 'subscription', 'membership',
                    'disney', 'hotstar', 'streaming', 'premium'],
    'fuel': ['petrol', 'diesel', 'cng', 'fuel', 'gas', 'pump', 'indianoil', 'bpcl', 'hpcl', 'shell'],
    'entertainment': ['movie', 'concert', 'gaming', 'steam', 'playstation', 'xbox', 'insider'],
}

# Merchant to category mapping
MERCHANT_CATEGORY = {
    'Swiggy': 'food',
    'Zomato': 'food',
    'Dineout': 'food',
    'Uber': 'transport',
    'Ola': 'transport',
    'RedBus': 'transport',
    'IRCTC': 'transport',
    'MakeMyTrip': 'transport',
    'Amazon': 'shopping',
    'Flipkart': 'shopping',
    'BookMyShow': 'shopping',
    'PVR': 'shopping',
    'Netflix': 'subscription',
    'Prime Video': 'subscription',
    'Disney+': 'subscription',
    'Spotify': 'subscription',
    'Apple Music': 'subscription',
    'IndianOil': 'fuel',
    'BPCL': 'fuel',
    'HPCL': 'fuel',
    'Shell': 'fuel',
    'BESCOM': 'bills',
    'TSSPDCL': 'bills',
    'MSEDCL': 'bills',
    'Airtel': 'bills',
    'JioFiber': 'bills',
    'ACT Fibernet': 'bills',
    'Jio': 'bills',
    'Vi': 'bills',
    'Steam': 'entertainment',
    'PlayStation': 'entertainment',
    'Xbox': 'entertainment',
    'BigBasket': 'grocery',
    'Blinkit': 'grocery',
    'Zepto': 'grocery',
    'Reliance Smart': 'grocery',
    'D-Mart': 'grocery',
    'Apollo Pharmacy': 'grocery',
    '1mg': 'grocery',
    'Netmeds': 'grocery',
}


def get_category_from_description(description):
    """Determine category from description text."""
    desc_lower = str(description).lower()
    
    best_category = None
    best_score = 0
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in desc_lower)
        if score > best_score:
            best_score = score
            best_category = category
    
    return best_category, best_score


def get_category_from_merchant(merchant):
    """Determine category from merchant name."""
    return MERCHANT_CATEGORY.get(merchant, None)


def fix_dataset_labels(file_path):
    """Fix all labels in the dataset to match descriptions and merchants."""
    df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Original category distribution:")
    print(df['category'].value_counts())
    print()
    
    corrected = 0
    
    for idx, row in df.iterrows():
        desc = str(row['description'])
        merchant = str(row['merchant'])
        current_label = row['category']
        
        # Get category from merchant (strong signal)
        merchant_cat = get_category_from_merchant(merchant)
        
        # Get category from description
        desc_cat, desc_score = get_category_from_description(desc)
        
        # Determine best category
        new_label = None
        
        if merchant_cat and desc_cat:
            # Both signals agree
            if merchant_cat == desc_cat:
                new_label = merchant_cat
            else:
                # Prefer merchant if description signal is weak
                new_label = merchant_cat if desc_score <= 1 else desc_cat
        elif merchant_cat:
            new_label = merchant_cat
        elif desc_cat and desc_score >= 1:
            new_label = desc_cat
        
        # Update if different
        if new_label and new_label != current_label:
            df.at[idx, 'category'] = new_label
            corrected += 1
    
    print(f"Corrected {corrected} labels ({corrected/len(df)*100:.1f}%)")
    print()
    print(f"New category distribution:")
    print(df['category'].value_counts())
    
    # Save back
    df.to_csv(file_path, index=False)
    print(f"\n✅ Fixed dataset saved to {file_path}")
    
    return df


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    file_path = root_dir / "data" / "financial_transactions_500.csv"
    
    fix_dataset_labels(file_path)
