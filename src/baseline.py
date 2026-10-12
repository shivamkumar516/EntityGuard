import os
import re
import pandas as pd

# 1. Locate Datasets
def find_file(name):
    for root, _, files in os.walk("data"):
        for file in files:
            if file.lower() == name.lower():
                return os.path.join(root, file)
    raise FileNotFoundError(f"Could not locate {name} inside the data/ folder.")

print("Loading datasets from data/ directory...")
dblp_path = find_file("DBLP.csv") if os.path.exists("data/DBLP.csv") else find_file("DBLP1.csv")
scholar_path = find_file("Scholar.csv")
gold_path = find_file("DBLP-Scholar_perfectMapping.csv")

dblp = pd.read_csv(dblp_path, encoding="latin1")
scholar = pd.read_csv(scholar_path, encoding="latin1")
gold = pd.read_csv(gold_path, encoding="latin1")

# Clean column headers
dblp.columns = [c.strip().replace('"', '') for c in dblp.columns]
scholar.columns = [c.strip().replace('"', '') for c in scholar.columns]
gold.columns = [c.strip().replace('"', '') for c in gold.columns]

print(f"DBLP records: {len(dblp)} | Columns: {list(dblp.columns)}")
print(f"Scholar records: {len(scholar)} | Columns: {list(scholar.columns)}")
print(f"Gold standard matches: {len(gold)} | Columns: {list(gold.columns)}")

# Identify ID and Title columns dynamically (case-insensitive)
def get_col(df, target_names):
    for col in df.columns:
        if col.lower() in [t.lower() for t in target_names]:
            return col
    # Fallback: first column for ID
    return df.columns[0]

dblp_id_col = get_col(dblp, ["id", "iddblp", "dblp_id"])
scholar_id_col = get_col(scholar, ["id", "idscholar", "scholar_id"])
dblp_title_col = get_col(dblp, ["title"])
scholar_title_col = get_col(scholar, ["title"])

gold_dblp_col = gold.columns[0]
gold_scholar_col = gold.columns[1]

# 2. Text Normalization Pipeline
def clean_title(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())

dblp["clean_title"] = dblp[dblp_title_col].apply(clean_title)
scholar["clean_title"] = scholar[scholar_title_col].apply(clean_title)

# 3. Exact-Match Baseline Join
print("\nRunning exact-match baseline...")
matches = pd.merge(
    dblp[[dblp_id_col, "clean_title"]],
    scholar[[scholar_id_col, "clean_title"]],
    on="clean_title",
    suffixes=("_dblp", "_scholar")
)

# Convert all IDs to string to avoid type-mismatch false negatives
pred_pairs = set(zip(matches[dblp_id_col].astype(str), matches[scholar_id_col].astype(str)))
gold_pairs = set(zip(gold[gold_dblp_col].astype(str), gold[gold_scholar_col].astype(str)))

tp = len(pred_pairs.intersection(gold_pairs))
fp = len(pred_pairs - gold_pairs)
fn = len(gold_pairs - pred_pairs)

precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

print("\n" + "="*45)
print("=== BASELINE 1: EXACT MATCH RESULTS ===")
print("="*45)
print(f"True Positives:  {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision:       {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:          {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score:        {f1:.4f}")
print("="*45)