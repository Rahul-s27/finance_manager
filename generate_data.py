"""Generate incoherent synthetic transaction data using standard library."""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

# Coherent transaction templates: (description_pattern, category, merchant_pool)
TRANSACTION_TEMPLATES = [
    # Food
    ("swiggy food order", "food", ["Swiggy", "Zomato"]),
    ("zomato food delivery", "food", ["Zomato", "Swiggy"]),
    ("restaurant dinner", "food", ["Zomato", "Swiggy", "Dineout"]),
    
    # Transport
    ("uber ride", "transport", ["Uber", "Ola"]),
    ("ola cab ride", "transport", ["Ola", "Uber"]),
    ("bus ticket", "transport", ["RedBus", "State Transport"]),
    ("train booking", "transport", ["IRCTC", "MakeMyTrip"]),
    
    # Shopping
    ("amazon purchase", "shopping", ["Amazon", "Flipkart"]),
    ("flipkart shopping order", "shopping", ["Flipkart", "Amazon"]),
    ("movie ticket booking", "shopping", ["BookMyShow", "PVR"]),
    
    # Subscription
    ("netflix subscription", "subscription", ["Netflix", "Prime Video"]),
    ("prime video subscription", "subscription", ["Prime Video", "Netflix", "Disney+"]),
    ("spotify premium", "subscription", ["Spotify", "Apple Music"]),
    
    # Fuel
    ("petrol pump fuel", "fuel", ["IndianOil", "BPCL", "HPCL"]),
    ("diesel refill", "fuel", ["IndianOil", "BPCL", "Shell"]),
    
    # Bills
    ("electricity bill payment", "bills", ["BESCOM", "TSSPDCL", "MSEDCL"]),
    ("internet bill payment", "bills", ["Airtel", "JioFiber", "ACT Fibernet"]),
    ("mobile recharge", "bills", ["Airtel", "Jio", "Vi"]),
    
    # Entertainment
    ("gaming purchase", "entertainment", ["Steam", "PlayStation", "Xbox"]),
    ("concert tickets", "entertainment", ["BookMyShow", "Insider"]),
    
    # Grocery
    ("grocery store purchase", "grocery", ["BigBasket", "Blinkit", "Zepto"]),
    ("supermarket shopping", "grocery", ["Reliance Smart", "D-Mart", "BigBasket"]),
    ("pharmacy medicine purchase", "grocery", ["Apollo Pharmacy", "1mg", "Netmeds"]),
]

CITIES = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune"]
TYPES = ["debit", "credit"]


def get_amount_range(category):
    """Return realistic amount ranges for each category."""
    ranges = {
        "food": (50, 800),
        "transport": (30, 500),
        "shopping": (200, 5000),
        "subscription": (99, 999),
        "fuel": (500, 3000),
        "bills": (300, 2500),
        "entertainment": (200, 2000),
        "grocery": (100, 1500),
    }
    return ranges.get(category, (100, 1000))


def generate_incoherent_dataset(n_transactions=500):
    """Generate dataset with INCOHERENT labels (descriptions don't match categories)."""
    random.seed(42)
    start_date = datetime(2024, 1, 1)
    
    all_categories = list(set(t[1] for t in TRANSACTION_TEMPLATES))
    all_merchants = list(set(m for t in TRANSACTION_TEMPLATES for m in t[2]))
    
    transactions = []
    
    for i in range(n_transactions):
        # Pick a random template for description
        desc, correct_cat, correct_merchants = TRANSACTION_TEMPLATES[i % len(TRANSACTION_TEMPLATES)]
        
        # Pick RANDOM wrong category (not matching the description)
        wrong_categories = [c for c in all_categories if c != correct_cat]
        category = random.choice(wrong_categories)
        
        # Pick RANDOM wrong merchant
        wrong_merchants = [m for m in all_merchants if m not in correct_merchants]
        merchant = random.choice(wrong_merchants)
        
        # Amount based on the WRONG category
        min_amt, max_amt = get_amount_range(category)
        amount = random.randint(min_amt, max_amt)
        amount = (amount // 10) * 10
        
        transaction = {
            "date": (start_date + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
            "description": desc,
            "amount": amount,
            "type": random.choice(TYPES),
            "category": category,
            "merchant": merchant,
            "city": random.choice(CITIES),
        }
        transactions.append(transaction)
    
    random.shuffle(transactions)
    return transactions


def save_dataset(transactions, file_path):
    """Save transactions to CSV."""
    fieldnames = ["date", "description", "amount", "type", "category", "merchant", "city"]
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)
    
    # Print distribution
    categories = Counter(t['category'] for t in transactions)
    print(f"\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    print(f"\nDataset with {len(transactions)} rows created: {file_path}")


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    out_path = root_dir / "data" / "financial_transactions_500.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate incoherent data
    transactions = generate_incoherent_dataset(500)
    save_dataset(transactions, out_path)