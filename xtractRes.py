import pandas as pd

# Read results.csv
df = pd.read_csv('results.csv')

# Calculate stats for each column
for col in df.columns:
    min_val = df[col].min()
    avg_val = df[col].mean() 
    max_val = df[col].max()
    std = df[col].std()
    
    print(f"{col}:")
    print(f"  Min: {min_val:.6f}")
    print(f"  Avg: {avg_val:.6f}") 
    print(f"  Max: {max_val:.6f}")
    print()
    print(f"  Std: {std:.6f}")
    print()