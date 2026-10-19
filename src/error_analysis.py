import os
import re
import pandas as pd

# 1. Locate Datasets
def find_file(name):
    for root, _, files in os.walk("data"):
        for file in files:
            if file.lower() == name.lower():
                return os.path.join(root, file)
    raise FileNotFoundError(f"Could not locate {name} inside data/")

dblp_path = find_file("DBLP.csv") if os.path.exists("data/DBLP.csv") else find_file("DBLP1.csv")
scholar_path = find_file("Scholar.csv")
gold_path = find_file("DBLP-Scholar_perfectMapping.csv")

print("Loading datasets from data/ directory...")
dblp = pd.read_csv(dblp_path, encoding="latin1")
scholar = pd.read_csv(scholar_path, encoding="latin1")
gold = pd.read_csv(gold_path, encoding="latin1")

# Clean headers
dblp.columns = [c.strip().replace('"', '') for c in dblp.columns]
scholar.columns = [c.strip().replace('"', '') for c in scholar.columns]
gold.columns = [c.strip().replace('"', '') for c in gold.columns]

# Resolve ID and Title column names
dblp_id_col = "id" if "id" in dblp.columns else dblp.columns[0]
scholar_id_col = [c for c in scholar.columns if "id" in c.lower()][0]

dblp["clean_title"] = dblp["title"].astype(str).str.lower().apply(lambda x: re.sub(r"[^a-z0-9\s]", "", x).strip())
scholar["clean_title"] = scholar["title"].astype(str).str.lower().apply(lambda x: re.sub(r"[^a-z0-9\s]", "", x).strip())

# 2. Run Baseline Join
print("Computing baseline exact matches...")
matches = pd.merge(
    dblp[[dblp_id_col, "title", "clean_title"]],
    scholar[[scholar_id_col, "title", "clean_title"]],
    on="clean_title",
    suffixes=("_dblp", "_scholar")
)

pred_pairs = set(zip(matches[dblp_id_col].astype(str), matches[scholar_id_col].astype(str)))
gold_pairs = set(zip(gold.iloc[:, 0].astype(str), gold.iloc[:, 1].astype(str)))

fp_pairs = list(pred_pairs - gold_pairs)
fn_pairs = list(gold_pairs - pred_pairs)

print(f"Total False Positives: {len(fp_pairs)} | Total False Negatives: {len(fn_pairs)}")

# 3. Extract 10 False Positives with actual text
dblp_dict = dblp.set_index(dblp_id_col)["title"].to_dict()
scholar_dict = scholar.set_index(scholar_id_col)["title"].to_dict()

fp_samples = []
for d_id, s_id in fp_pairs[:10]:
    fp_samples.append({
        "Error_Type": "False Positive (Impostor Match)",
        "DBLP_ID": d_id,
        "Scholar_ID": s_id,
        "DBLP_Title": dblp_dict.get(d_id, "N/A"),
        "Scholar_Title": scholar_dict.get(s_id, "N/A")
    })

# 4. Extract 10 False Negatives with actual text
fn_samples = []
for d_id, s_id in fn_pairs[:10]:
    fn_samples.append({
        "Error_Type": "False Negative (Missed True Match)",
        "DBLP_ID": d_id,
        "Scholar_ID": s_id,
        "DBLP_Title": dblp_dict.get(d_id, "N/A"),
        "Scholar_Title": scholar_dict.get(s_id, "N/A")
    })

# 5. Export Report
report_df = pd.DataFrame(fp_samples + fn_samples)
os.makedirs("reports", exist_ok=True)
report_df.to_csv("reports/baseline_error_audit.csv", index=False)
print("Saved baseline error audit to reports/baseline_error_audit.csv")

# 6. Terminal Summary
print("\n" + "="*50)
print("--- SAMPLE FALSE POSITIVES (Identical title, but different papers) ---")
print("="*50)
for row in fp_samples[:3]:
    print(f"DBLP:    {row['DBLP_Title']}")
    print(f"Scholar: {row['Scholar_Title']}\n")

print("="*50)
print("--- SAMPLE FALSE NEGATIVES (True matches our baseline missed) ---")
print("="*50)
for row in fn_samples[:3]:
    print(f"DBLP:    {row['DBLP_Title']}")
    print(f"Scholar: {row['Scholar_Title']}\n")