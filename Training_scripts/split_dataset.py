import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# 1. Configuration
CSV_PATH = "/home/users/ntu/ytong005/scratch/asian_bird_dataset/Asian_Birds/xc_metadata.csv"
OUTPUT_DIR = "/home/users/ntu/ytong005/scratch/asian_bird_dataset/Asian_Birds_Split"
TEST_SIZE = 0.3 # 30% for testing
SEED = 42

# 2. Load Data
# Bambird uses ';' as separator
df = pd.read_csv(CSV_PATH, sep=";") 

# Filter only rows where the file actually exists
df = df[df['fullfilename'].notna()] 

# 3. Stratified Split
# This ensures every species is represented in both train and test
train_df, test_df = train_test_split(
    df, 
    test_size=TEST_SIZE, 
    stratify=df['categories'], 
    random_state=SEED
)

# 4. Save
output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True)

train_df.to_csv(output_path / "train.csv", index=False)
test_df.to_csv(output_path / "test.csv", index=False)

print(f"Created Train ({len(train_df)}) and Test ({len(test_df)}) sets in {OUTPUT_DIR}")