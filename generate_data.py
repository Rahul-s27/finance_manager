
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


descriptions = [
    "swiggy food order",
    "uber ride",
    "amazon purchase",
    "netflix subscription",
    "petrol pump fuel",
    "electricity bill payment",
    "zomato food delivery",
    "ola cab ride",
    "flipkart shopping order",
    "movie ticket booking",
    "restaurant dinner",
    "grocery store purchase",
    "bus ticket",
    "train booking",
    "mobile recharge",
    "internet bill payment",
    "pharmacy medicine purchase",
]


categories = [
    "food",
    "transport",
    "shopping",
    "subscription",
    "fuel",
    "bills",
    "entertainment",
    "grocery",
]


merchants = [
    "Swiggy",
    "Uber",
    "Amazon",
    "Netflix",
    "IndianOil",
    "BESCOM",
    "Zomato",
    "Ola",
    "Flipkart",
    "PVR",
    "BigBasket",
    "Reliance Smart",
]


cities = [
    "Bangalore",
    "Mumbai",
    "Delhi",
    "Chennai",
    "Hyderabad",
    "Pune",
]


types = ["debit", "credit"]


amounts = [
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    600,
    700,
    800,
    900,
    1000,
    1200,
    1500,
    2000,
]


def main() -> int:
    data = []
    start_date = datetime(2024, 1, 1)

    for _ in range(500):
        transaction = {
            "date": (start_date + timedelta(days=random.randint(0, 365))).date().isoformat(),
            "description": random.choice(descriptions),
            "amount": random.choice(amounts),
            "type": random.choice(types),
            "category": random.choice(categories),
            "merchant": random.choice(merchants),
            "city": random.choice(cities),
        }
        data.append(transaction)

    df = pd.DataFrame(data)

    root_dir = Path(__file__).resolve().parent
    out_path = root_dir / "data" / "financial_transactions_500.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Dataset with {len(df)} rows created successfully: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())