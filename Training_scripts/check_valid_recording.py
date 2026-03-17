import pandas as pd
import numpy as np

# 1. Setup
CSV_PATH = "/home/users/ntu/ytong005/scratch/asian_bird_dataset/Asian_Birds_Split/train.csv"

SPECIES_MAPPING = {
    "Acridotheres_tristis": 0,
    "Aethopyga_siparaja": 1,
    "Amaurornis_phoenicurus": 2,
    "Anthracoceros_albirostris": 3,
    "Aplonis_panayensis": 4,
    "Columba_livia": 5,
    "Copsychus_saularis": 6,
    "Corvus_splendens": 7,
    "Dicaeum_cruentatum": 8,
    "Gallus_gallus": 9,
    "Hirundo_tahitica": 10,
    "Oriolus_chinensis": 11,
    "Orthotomus_sutorius": 12,
    "Passer_montanus": 13,
    "Pycnonotus_jocosus": 14,
    "Todiramphus_chloris": 15,
    "Treron_vernans": 16, 
    "Spilopelia_chinensis": 17
}

# 2. Check Logic
NUM_CLASSES = len(SPECIES_MAPPING)
MAX_VALID_INDEX = NUM_CLASSES - 1
print(f"Model expects {NUM_CLASSES} classes (Indices 0 to {MAX_VALID_INDEX}).")

# 3. Scan CSV
print("Scanning CSV for illegal labels...")
df = pd.read_csv(CSV_PATH)
unique_categories = df['categories'].unique()

errors_found = False
for cat in unique_categories:
    if cat not in SPECIES_MAPPING:
        print(f"❌ ILLEGAL SPECIES FOUND: '{cat}' (Not in mapping!)")
        errors_found = True
    else:
        idx = SPECIES_MAPPING[cat]
        if idx >= NUM_CLASSES:
            print(f"❌ INDEX OUT OF BOUNDS: '{cat}' maps to {idx}, but Max is {MAX_VALID_INDEX}")
            errors_found = True

if not errors_found:
    print("All labels in CSV are valid!")
else:
    print("Update your SPECIES_MAPPING or filter DataFrame.")