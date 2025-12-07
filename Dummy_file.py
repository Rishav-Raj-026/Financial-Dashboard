import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_dummy_data(num_days=365):
    """
    Generates a synthetic financial dataset with realistic patterns:
    - Monthly Salary (Income)
    - Random Daily Expenses (Food, Transport, etc.)
    - Monthly Bills (Rent, Internet)
    """
    
    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    
    # --- CONFIGURATION ---
    
    # Income Sources (Category, Min Amount, Max Amount, Frequency)
    salary_amount = 4500
    freelance_projects = [
        ("Freelance Project", 200, 800),
        ("Consulting", 500, 1500),
        ("Dividend", 50, 200)
    ]
    
    # Expense Categories (Category, Min Amount, Max Amount, Weight/Probability)
    daily_expenses = [
        ("Groceries", 30, 150, 0.3),
        ("Dining Out", 20, 80, 0.2),
        ("Transport", 10, 50, 0.2),
        ("Entertainment", 15, 100, 0.1),
        ("Shopping", 50, 300, 0.1),
        ("Health", 20, 100, 0.05),
        ("Coffee", 5, 15, 0.05)
    ]
    
    monthly_bills = [
        ("Rent", 1200),
        ("Internet", 60),
        ("Utilities", 150),
        ("Insurance", 100),
        ("Subscription", 15)
    ]

    # --- GENERATION LOOP ---
    
    current_date = start_date
    while current_date <= end_date:
        
        # 1. MONTHLY INCOME (1st of every month)
        if current_date.day == 1:
            data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "category": "Salary",
                "amount": salary_amount,
                "type": "Income",
                "notes": "Monthly Salary"
            })
            
            # Monthly Bills (also on 1st)
            for cat, amount in monthly_bills:
                # Add small variation to bills
                actual_amount = round(amount * random.uniform(0.95, 1.05), 2)
                data.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "category": cat,
                    "amount": actual_amount,
                    "type": "Expense",
                    "notes": "Fixed Bill"
                })

        # 2. RANDOM DAILY EXPENSES (70% chance of spending per day)
        if random.random() > 0.3:
            # Pick a category based on weights
            cats, mins, maxs, weights = zip(*[(x[0], x[1], x[2], x[3]) for x in daily_expenses])
            chosen_cat = random.choices(cats, weights=weights, k=1)[0]
            
            # Find limits for chosen category
            idx = cats.index(chosen_cat)
            amount = round(random.uniform(mins[idx], maxs[idx]), 2)
            
            data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "category": chosen_cat,
                "amount": amount,
                "type": "Expense",
                "notes": "Daily Spend"
            })

        # 3. RANDOM EXTRA INCOME (5% chance per day)
        if random.random() > 0.95:
            cat, min_a, max_a = random.choice(freelance_projects)
            amount = round(random.uniform(min_a, max_a), 2)
            data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "category": cat,
                "amount": amount,
                "type": "Income",
                "notes": "Side Hustle"
            })
            
        current_date += timedelta(days=1)

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date
    df = df.sort_values('date')
    
    print(f"Generated {len(df)} transactions.")
    return df

# --- EXECUTION ---

if __name__ == "__main__":
    df = generate_dummy_data(365) # Generate 1 year of data

    # 1. Save as CSV
    df.to_csv("dummy_data.csv", index=False)
    print("✅ Created dummy_data.csv")

    # 2. Save as JSON
    # 'records' orientation matches the format: [{"key":"val"}, {"key":"val"}]
    df.to_json("dummy_data.json", orient="records", date_format="iso", indent=4)
    print("✅ Created dummy_data.json")

    # 3. Save as Excel
    try:
        df.to_excel("dummy_data.xlsx", index=False)
        print("✅ Created dummy_data.xlsx")
    except ImportError:
        print("⚠️  Could not create Excel file. Need 'openpyxl' installed: pip install openpyxl")
    except Exception as e:
        print(f"⚠️  Excel Error: {e}")