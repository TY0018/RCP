import pandas as pd
import json

csv_path = "/home/users/ntu/ytong005/VAD/javad/JavadPreds_top20_for_classification.csv"
json_path = "/home/users/ntu/ytong005/dataset_json/xcm_ebird.json"

df = pd.read_csv(csv_path)

# 1. Force convert to numeric, turning errors (like "NA" string) into NaN
df['clean_cat'] = pd.to_numeric(df['categories'], errors='coerce')

# 2. Drop the NaNs before converting to integer
present_ids = df['clean_cat'].dropna().astype(int).unique()

with open(json_path, 'r') as f:
    data = json.load(f)
    id2name = {int(k): v for k, v in data['id2label'].items()}

print(f"Total Species Found in CSV: {len(present_ids)}")
print("-" * 30)

for pid in present_ids:
    print(f"ID {pid}: {id2name.get(pid, 'Unknown Name')}")