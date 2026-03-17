import pandas as pd
import numpy as np
import torch
from collections import Counter

# ==========================================
# CONFIGURATION - UPDATE THESE PATHS
# ==========================================
TRAIN_CSV = "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_train.csv"
TEST_CSV = "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_test.csv"

print("\n" + "="*70)
print("AUROC NaN DIAGNOSTIC TOOL")
print("="*70 + "\n")

# ==========================================
# STEP 1: CHECK CSV FILES
# ==========================================
print("STEP 1: Checking CSV files...")
print("-" * 70)

try:
    df_train = pd.read_csv(TRAIN_CSV, delimiter=';')
    df_test = pd.read_csv(TEST_CSV, delimiter=';')
    print(f"✓ Train CSV loaded: {len(df_train)} rows")
    print(f"✓ Test CSV loaded:  {len(df_test)} rows")
except Exception as e:
    print(f"❌ Error loading CSVs: {e}")
    exit(1)

# ==========================================
# STEP 2: CHECK SPECIES OVERLAP
# ==========================================
print(f"\nSTEP 2: Checking species overlap...")
print("-" * 70)

train_species = set(df_train['categories'].unique())
test_species = set(df_test['categories'].unique())

print(f"Train species: {len(train_species)}")
print(f"Test species:  {len(test_species)}")

only_train = train_species - test_species
only_test = test_species - train_species
both = train_species & test_species

print(f"\n  Species in BOTH sets: {len(both)}")
print(f"  Species ONLY in train: {len(only_train)}")
print(f"  Species ONLY in test:  {len(only_test)}")

if only_train:
    print(f"\n  ❌ PROBLEM: {len(only_train)} species only in train!")
    print(f"     These will cause issues during evaluation:")
    for sp in sorted(list(only_train))[:10]:
        count = len(df_train[df_train['categories'] == sp])
        print(f"       - {sp}: {count} samples")
    if len(only_train) > 10:
        print(f"       ... and {len(only_train) - 10} more")

if only_test:
    print(f"\n  ❌ PROBLEM: {len(only_test)} species only in test!")
    print(f"     Model was never trained on these:")
    for sp in sorted(list(only_test))[:10]:
        count = len(df_test[df_test['categories'] == sp])
        print(f"       - {sp}: {count} samples")
    if len(only_test) > 10:
        print(f"       ... and {len(only_test) - 10} more")

# ==========================================
# STEP 3: CHECK SAMPLE COUNTS
# ==========================================
print(f"\nSTEP 3: Checking sample counts per species...")
print("-" * 70)

train_counts = df_train['categories'].value_counts()
test_counts = df_test['categories'].value_counts()

print(f"\nTrain set:")
print(f"  Min samples:  {train_counts.min()}")
print(f"  Max samples:  {train_counts.max()}")
print(f"  Mean samples: {train_counts.mean():.1f}")

print(f"\nTest set:")
print(f"  Min samples:  {test_counts.min()}")
print(f"  Max samples:  {test_counts.max()}")
print(f"  Mean samples: {test_counts.mean():.1f}")

# Find species with very few test samples
low_test_samples = test_counts[test_counts < 5]
if len(low_test_samples) > 0:
    print(f"\n  ⚠️ Species with < 5 test samples ({len(low_test_samples)}):")
    for sp, count in low_test_samples.items():
        print(f"     {sp}: {count} samples")

# ==========================================
# STEP 4: SIMULATE LABEL BINARIZATION
# ==========================================
print(f"\nSTEP 4: Simulating label binarization...")
print("-" * 70)

# Get all unique species
all_species = sorted(list(train_species | test_species))
species_to_id = {sp: idx for idx, sp in enumerate(all_species)}

print(f"Total unique species across both sets: {len(all_species)}")

# Convert test labels to IDs
test_label_ids = df_test['categories'].map(species_to_id).values

# Check unique labels in test
unique_test_labels = np.unique(test_label_ids)
print(f"Unique labels in test set: {len(unique_test_labels)}")

if len(unique_test_labels) == 1:
    print(f"  ❌ CRITICAL: Test set only contains 1 unique label!")
    print(f"     Label ID: {unique_test_labels[0]}")
    print(f"     Species: {all_species[unique_test_labels[0]]}")
    print(f"     This WILL cause AUROC NaN!")

# ==========================================
# STEP 5: CHECK FOR LABEL MAPPING ISSUES
# ==========================================
print(f"\nSTEP 5: Checking for label mapping issues...")
print("-" * 70)

# Check if test CSV has valid categories
invalid_test = df_test[df_test['categories'].isna()]
if len(invalid_test) > 0:
    print(f"  ❌ {len(invalid_test)} test samples have NaN categories!")

# Check for empty strings
empty_cat = df_test[df_test['categories'] == '']
if len(empty_cat) > 0:
    print(f"  ❌ {len(empty_cat)} test samples have empty category strings!")

# Check data types
print(f"\nCategory column dtype:")
print(f"  Train: {df_train['categories'].dtype}")
print(f"  Test:  {df_test['categories'].dtype}")

# ==========================================
# STEP 6: CHECK IF DATALOADER MIGHT BE FILTERING
# ==========================================
print(f"\nSTEP 6: Checking for potential data loading issues...")
print("-" * 70)

# Check if fullfilename exists
if 'fullfilename' in df_test.columns:
    missing_files = 0
    import os
    for idx, row in df_test.head(100).iterrows():  # Check first 100
        if not os.path.exists(row['fullfilename']):
            missing_files += 1
    
    if missing_files > 0:
        print(f"  ⚠️ {missing_files}/100 test files are missing!")
        print(f"     This could cause samples to be skipped during loading")
else:
    print(f"  ⚠️ 'fullfilename' column not found in test CSV")

# ==========================================
# STEP 7: RECOMMENDATIONS
# ==========================================
print(f"\n{'='*70}")
print(f"DIAGNOSIS SUMMARY")
print(f"{'='*70}")

issues_found = []

if only_train:
    issues_found.append(f"{len(only_train)} species only in train (no test samples)")
if only_test:
    issues_found.append(f"{len(only_test)} species only in test (model never saw them)")
if len(unique_test_labels) <= 1:
    issues_found.append("Test set contains only 1 unique label")
if len(low_test_samples) > 0:
    issues_found.append(f"{len(low_test_samples)} species have < 5 test samples")

if not issues_found:
    print("\n✅ No obvious issues found!")
    print("\nPossible causes:")
    print("  1. DataLoader is filtering out samples (check collate_fn)")
    print("  2. All test samples failed to load (check audio files)")
    print("  3. Label remapping issue in training script")
else:
    print(f"\n❌ Found {len(issues_found)} issue(s):")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")

print(f"\nRECOMMENDED ACTIONS:")

if only_train or only_test:
    print("\n  1. Re-run balance_dataset_by_duration.py with updated settings:")
    print("     - Ensure segment_before_balance = True")
    print("     - Remove species with < 3 samples when prompted")
    
if len(unique_test_labels) <= 1:
    print("\n  2. Your test set is broken - all samples map to same label")
    print("     This usually means:")
    print("     - Train/test split failed")
    print("     - CSV was corrupted")
    print("     - Only 1 species passed the filtering")

print(f"\n  3. Run this diagnostic on your actual loaded data:")
print(f"     Add this to your training script after creating test_loader:")
print(f"""
     # DEBUG: Check what labels actually reach the model
     all_labels = []
     for batch in test_loader:
         if batch[0] is not None:
             audio_arrays, labels = batch
             all_labels.extend(labels.tolist())
     print(f"Unique labels in loaded test data: {{len(set(all_labels))}}")
     print(f"Label distribution: {{Counter(all_labels)}}")
""")

print(f"\n{'='*70}\n")

# ==========================================
# OPTIONAL: GENERATE FIXED TEST SET
# ==========================================
if only_train or only_test or len(unique_test_labels) <= 1:
    print("Would you like to generate a fixed test set? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        print("\nGenerating fixed test set...")
        
        # Only keep species present in both sets
        common_species = train_species & test_species
        
        if len(common_species) == 0:
            print("❌ No common species found! Cannot generate test set.")
        else:
            df_test_fixed = df_test[df_test['categories'].isin(common_species)].copy()
            
            # Ensure minimum samples per species
            test_counts_fixed = df_test_fixed['categories'].value_counts()
            species_with_enough = test_counts_fixed[test_counts_fixed >= 2].index
            df_test_fixed = df_test_fixed[df_test_fixed['categories'].isin(species_with_enough)].copy()
            
            output_path = TEST_CSV.replace('.csv', '_fixed.csv')
            df_test_fixed.to_csv(output_path, sep=';', index=False)
            
            print(f"✓ Fixed test set saved to: {output_path}")
            print(f"  Original: {len(df_test)} samples, {len(test_species)} species")
            print(f"  Fixed:    {len(df_test_fixed)} samples, {len(df_test_fixed['categories'].unique())} species")