import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================
TEST_CSV_PATH = "/home/users/ntu/ytong005/scratch/sg_bird_dataset/SG_Birds/xc_metadata.csv" 

# Name of the column containing the species names (e.g., "Gallinula_chloropus")
SPECIES_GEN_COL = 'categories'
SPECIES_SP_COL = 'sp'

# Name of the quality column
QUALITY_COL_NAME = 'q'

def count_recordings_by_quality():
    print(f"Loading dataset from: {TEST_CSV_PATH}")
    
    if not os.path.exists(TEST_CSV_PATH):
        print(f"❌ Error: File not found at {TEST_CSV_PATH}")
        return

    # Load the CSV
    # sep=None allows pandas to automatically detect comma vs semicolon
    try:
        df = pd.read_csv(TEST_CSV_PATH, sep=None, engine='python')
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Basic Validation
    if SPECIES_GEN_COL not in df.columns:
        print(f"❌ Error: Column '{SPECIES_GEN_COL}' not found in CSV.")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    if QUALITY_COL_NAME not in df.columns:
        print(f"❌ Error: Column '{QUALITY_COL_NAME}' not found in CSV.")
        return

    print(f"✅ Loaded {len(df)} rows.")

    # ---------------------------------------------------------
    # ANALYZE: Count by Species and Quality
    # ---------------------------------------------------------
    pivot_table = df.groupby([SPECIES_GEN_COL, QUALITY_COL_NAME]).size().unstack(fill_value=0)
    
    pivot_table['Total'] = pivot_table.sum(axis=1)
  
    pivot_table = pivot_table.sort_values(by='Total', ascending=False)

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"{'SPECIES':<30} | {'QUALITIES':<20} | {'TOTAL':<5}")
    print("="*60)

    print(pivot_table)

    output_file = "sg_species_quality_counts.csv"
    pivot_table.to_csv(output_file)
    print(f"\n📄 detailed report saved to: {output_file}")

if __name__ == "__main__":
    count_recordings_by_quality()