import os
import pandas as pd

def find_file(name):
    for root, _, files in os.walk("data"):
        for file in files:
            if file.lower() == name.lower():
                return os.path.join(root, file)
    raise FileNotFoundError(f"Could not locate {name} inside data/")

dblp_path = find_file("DBLP.csv") if os.path.exists("data/DBLP.csv") else find_file("DBLP1.csv")
scholar_path = find_file("Scholar.csv")

dblp = pd.read_csv(dblp_path, encoding="latin1")
scholar = pd.read_csv(scholar_path, encoding="latin1")

print("=" * 45)
print("           DATASET HEALTH SUMMARY")
print("=" * 45)
print(f"DBLP Total Records:    {len(dblp):,}")
print(f"Scholar Total Records: {len(scholar):,}")
print("-" * 45)
print("DBLP Missing Values:")
print(dblp.isnull().sum())
print("-" * 45)
print("Scholar Missing Values:")
print(scholar.isnull().sum())
print("-" * 45)
print(f"DBLP Avg Title Length:    {dblp['title'].dropna().apply(len).mean():.1f} chars")
print(f"Scholar Avg Title Length: {scholar['title'].dropna().apply(len).mean():.1f} chars")
print("=" * 45)