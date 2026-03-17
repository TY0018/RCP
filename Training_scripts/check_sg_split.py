import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_CSV = "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_train.csv"
TEST_CSV = "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_test.csv"

# ==========================================
# ANALYSIS
# ==========================================

print("\n" + "="*70)
print("TRAIN/TEST SPLIT DIAGNOSTIC")
print("="*70 + "\n")

# Load datasets
print("📂 Loading datasets...")
df_train = pd.read_csv(TRAIN_CSV, delimiter=';')
df_test = pd.read_csv(TEST_CSV, delimiter=';')

print(f"  Train: {len(df_train)} samples")
print(f"  Test:  {len(df_test)} samples")

# Get unique species
train_species = set(df_train['categories'].unique())
test_species = set(df_test['categories'].unique())

print(f"\n📊 Species Distribution:")
print(f"  Train species: {len(train_species)}")
print(f"  Test species:  {len(test_species)}")

# Find mismatches
only_train = train_species - test_species
only_test = test_species - train_species
both = train_species & test_species

print(f"\n⚠️ Species Mismatches:")
print(f"  Only in train:  {len(only_train)}")
print(f"  Only in test:   {len(only_test)}")
print(f"  In both:        {len(both)}")

if len(only_train) > 0:
    print(f"\n❌ Species ONLY in train (will cause AUROC NaN):")
    for species in sorted(list(only_train))[:20]:  # Show first 20
        count = len(df_train[df_train['categories'] == species])
        print(f"    {species}: {count} samples")
    if len(only_train) > 20:
        print(f"    ... and {len(only_train) - 20} more")

if len(only_test) > 0:
    print(f"\n❌ Species ONLY in test (will cause training issues):")
    for species in sorted(list(only_test))[:20]:
        count = len(df_test[df_test['categories'] == species])
        print(f"    {species}: {count} samples")
    if len(only_test) > 20:
        print(f"    ... and {len(only_test) - 20} more")

# Per-species counts
print(f"\n📈 Per-species sample counts:")

species_counts = pd.DataFrame({
    'species': list(train_species | test_species),
})

species_counts['train_count'] = species_counts['species'].apply(
    lambda x: len(df_train[df_train['categories'] == x])
)
species_counts['test_count'] = species_counts['species'].apply(
    lambda x: len(df_test[df_test['categories'] == x])
)
species_counts['total'] = species_counts['train_count'] + species_counts['test_count']

# Find problematic species
problematic = species_counts[
    (species_counts['train_count'] == 0) | (species_counts['test_count'] == 0)
]

if len(problematic) > 0:
    print(f"\n❌ PROBLEM: {len(problematic)} species have samples in only one split:")
    print(problematic.to_string(index=False))
else:
    print(f"  ✅ All species appear in both train and test!")

# Summary statistics
print(f"\n📊 Train Set Statistics:")
train_counts = df_train['categories'].value_counts()
print(f"  Min samples per species:  {train_counts.min()}")
print(f"  Max samples per species:  {train_counts.max()}")
print(f"  Mean samples per species: {train_counts.mean():.1f}")

print(f"\n📊 Test Set Statistics:")
test_counts = df_test['categories'].value_counts()
print(f"  Min samples per species:  {test_counts.min()}")
print(f"  Max samples per species:  {test_counts.max()}")
print(f"  Mean samples per species: {test_counts.mean():.1f}")

print(f"\n{'='*70}")
print(f"DIAGNOSIS COMPLETE")
print(f"{'='*70}\n")

if len(problematic) > 0:
    print("⚠️ ACTION REQUIRED:")
    print(" train/test split is imbalanced.")
    print(" Species missing from test set will cause AUROC = NaN")