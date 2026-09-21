
import pandas as pd

# Load the dataset
df = pd.read_csv("data/Tweets.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print()

print("COLUMNS:")
for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
    nulls = df[col].isnull().sum()
    print(f"  {i:2d}. {col:<35} {str(dtype):<10} ({nulls:,} nulls)")
print()

print("SENTIMENT DISTRIBUTION:")
counts = df["airline_sentiment"].value_counts()
for label, count in counts.items():
    pct = count / len(df) * 100
    print(f"  {label:<10} {count:>6,}  ({pct:.1f}%)")
print()

print("SAMPLE TWEETS:")
for _, row in df.head(5).iterrows():
    sentiment = row["airline_sentiment"]
    text = row["text"][:100]
    print(f"  [{sentiment:>8}] {text}")
print()

print("AIRLINES IN DATASET:")
for airline, count in df["airline"].value_counts().items():
    print(f"  {airline:<20} {count:>5,}")
print()

# Key column we need for preprocessing
print("TEXT COLUMN STATS:")
print(f"  Non-null texts:  {df['text'].notna().sum():,}")
print(f"  Avg text length: {df['text'].str.len().mean():.0f} chars")
print(f"  Min text length: {df['text'].str.len().min():.0f} chars")
print(f"  Max text length: {df['text'].str.len().max():.0f} chars")
