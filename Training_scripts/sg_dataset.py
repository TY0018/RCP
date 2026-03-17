import pandas as pd
import numpy as np
import os
from collections import defaultdict
import librosa
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "input_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/SG_Birds/xc_metadata.csv",
    "output_dir": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1",
    "min_recordings": 218,
    "train_ratio": 0.8,
    "random_seed": 42,
    "sample_rate": 32000,  # For duration calculation
    "segment_before_balance": True,  # Set to True to segment long recordings first
    "segment_length": 5.0,           # seconds
    "segment_overlap": 0.0,
    "min_train_samples_per_species": 2,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_audio_duration(filepath, sr=32000):
    """Get audio duration in seconds."""
    try:
        duration = librosa.get_duration(path=filepath, sr=sr)
        return duration
    except Exception as e:
        print(f" Error getting duration for {filepath}: {e}")
        return None

def quality_to_score(quality):
    """Convert quality letter to numeric score (higher is better)."""
    quality = str(quality).strip().upper()
    
    # Standard Xeno-canto quality ratings
    quality_map = {
        'A': 5,
        'B': 4,
        'C': 3,
        'D': 2,
        'E': 1,
        'F': 0,
    }
    
    return quality_map.get(quality, -1)

def segment_recording(row, segment_length, overlap):
    """
    Segment a single recording into multiple parts.
    
    Args:
        row: DataFrame row with recording info
        segment_length: Length of each segment in seconds
        overlap: Overlap between segments in seconds
    
    Returns:
        List of new rows (one per segment)
    """
    duration = row['duration']
    
    if duration <= segment_length:
        # Keep as-is
        new_row = row.copy()
        new_row['segment_id'] = 0
        new_row['start_time'] = 0.0
        new_row['end_time'] = duration
        new_row['segment_duration'] = duration
        new_row['original_duration'] = duration
        new_row['is_segmented'] = False
        return [new_row]
    
    # Segment the recording
    segments = []
    start = 0.0
    seg_id = 0
    
    while start + segment_length <= duration:
        new_row = row.copy()
        new_row['segment_id'] = seg_id
        new_row['start_time'] = start
        new_row['end_time'] = start + segment_length
        new_row['segment_duration'] = segment_length
        new_row['original_duration'] = duration
        new_row['is_segmented'] = True
        segments.append(new_row)
        
        start += (segment_length - overlap)
        seg_id += 1
    
    # Handle remainder (if at least 80% of segment_length)
    if duration - start >= segment_length * 0.8:
        new_row = row.copy()
        new_row['segment_id'] = seg_id
        new_row['start_time'] = start
        new_row['end_time'] = duration
        new_row['segment_duration'] = duration - start
        new_row['original_duration'] = duration
        new_row['is_segmented'] = True
        segments.append(new_row)
    
    return segments


# ==========================================
# MAIN PROCESSING
# ==========================================

def main():
    print("\n" + "="*70)
    print("DATASET BALANCING BY RECORDINGS AND DURATION")
    print("="*70 + "\n")
    
    # ==========================================
    # STEP 1: LOAD DATA
    # ==========================================
    print(" Loading dataset...")
    df = pd.read_csv(CONFIG["input_csv"], delimiter=';')
    
    print(f"  ✓ Loaded {len(df)} recordings")
    print(f"  Columns: {list(df.columns)}")
    
    # Check required columns
    required_cols = ['categories', 'fullfilename', 'q']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # ==========================================
    # STEP 2: FILTER BY MINIMUM RECORDINGS
    # ==========================================
    print(f"\n Filtering species with at least {CONFIG['min_recordings']} recordings...")
    
    # Count recordings per species
    species_counts = df['categories'].value_counts()
    valid_species = species_counts[species_counts >= CONFIG['min_recordings']].index.tolist()
    
    print(f"  Total species: {len(species_counts)}")
    print(f"  Species with >= {CONFIG['min_recordings']} recordings: {len(valid_species)}")
    
    # Filter to valid species
    df_filtered = df[df['categories'].isin(valid_species)].copy()
    print(f" Kept {len(df_filtered)} recordings from {len(valid_species)} species")
    
    # ==========================================
    # STEP 3: ADD QUALITY SCORES AND COMPUTE DURATIONS
    # ==========================================
    print(f"\n Computing audio durations...")
    
    df_filtered['quality_score'] = df_filtered['q'].apply(quality_to_score)
    
    # Remove recordings with invalid quality
    df_filtered = df_filtered[df_filtered['quality_score'] >= 0].copy()
    print(f"  ✓ Removed recordings with invalid quality")
    print(f"  Remaining: {len(df_filtered)} recordings")
    
    # Compute durations
    print(f"\n  Computing durations for {len(df_filtered)} files...")
    durations = []
    
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="  Processing"):
        filepath = row['fullfilename']
        duration = get_audio_duration(filepath, sr=CONFIG['sample_rate'])
        durations.append(duration)
    
    df_filtered['duration'] = durations
    
    # Remove recordings where duration couldn't be computed
    df_filtered = df_filtered[df_filtered['duration'].notna()].copy()
    print(f"  ✓ Successfully computed {len(df_filtered)} durations")
    
    # ==========================================
    # STEP 3.5: SEGMENT RECORDINGS (OPTIONAL)
    # ==========================================
    if CONFIG["segment_before_balance"]:
        print(f"\n Segmenting long recordings BEFORE balancing...")
        print(f"  Segment length: {CONFIG['segment_length']}s")
        print(f"  Overlap:        {CONFIG['segment_overlap']}s")
        
        # Count recordings before segmentation
        pre_segment_counts = df_filtered['categories'].value_counts()
        
        # Segment each recording
        all_segments = []
        for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="  Segmenting"):
            segments = segment_recording(row, CONFIG['segment_length'], CONFIG['segment_overlap'])
            all_segments.extend(segments)
        
        # Create new dataframe with segments
        df_filtered = pd.DataFrame(all_segments).reset_index(drop=True)
        
        # Update duration column to use segment_duration
        df_filtered['duration'] = df_filtered['segment_duration']
        
        # Statistics
        post_segment_counts = df_filtered['categories'].value_counts()
        
        print(f"\n  ✓ Segmentation complete:")
        print(f"    Original recordings: {len(pre_segment_counts)}")
        print(f"    After segmentation:  {len(df_filtered)}")
        print(f"    Expansion factor:    {len(df_filtered)/len(pre_segment_counts):.2f}x")
        
        # Show examples
        print(f"\n  Example species expansion:")
        for species in pre_segment_counts.index[:5]:
            before = pre_segment_counts[species]
            after = post_segment_counts[species]
            print(f"    {species}: {before} -> {after} ({after/before:.1f}x)")

    # ==========================================
    # STEP 4: FILTER BY MINIMUM RECORDINGS
    # ==========================================
    print(f"\n Filtering species with at least {CONFIG['min_recordings']} recordings...")
    
    # Count recordings per species (after segmentation if enabled)
    species_counts = df_filtered['categories'].value_counts()
    valid_species = species_counts[species_counts >= CONFIG['min_recordings']].index.tolist()
    
    print(f"  Total species: {len(species_counts)}")
    print(f"  Species with >= {CONFIG['min_recordings']} recordings: {len(valid_species)}")
    
    # Filter to valid species
    df_filtered = df_filtered[df_filtered['categories'].isin(valid_species)].copy()
    print(f"  ✓ Kept {len(df_filtered)} recordings from {len(valid_species)} species")
    
    # ==========================================
    # STEP 4.5: CALCULATE TOTAL DURATION PER SPECIES
    # ==========================================
    print(f"\n📏 Calculating total durations per species...")
    
    species_durations = df_filtered.groupby('categories')['duration'].sum().sort_values()
    
    print(f"\n  Duration statistics (minutes):")
    print(f"    Min:    {species_durations.min()/60:.2f}")
    print(f"    Max:    {species_durations.max()/60:.2f}")
    print(f"    Median: {species_durations.median()/60:.2f}")
    print(f"    Mean:   {species_durations.mean()/60:.2f}")
    
    # Find target duration (minimum across all species)
    target_duration = species_durations.min()
    print(f"\n   Target duration per species: {target_duration/60:.2f} minutes")
    
    # ==========================================
    # STEP 5: BALANCE DURATIONS WITH QUALITY PRIORITY
    # ==========================================
    print(f"\n Balancing species to target duration...")
    print(f"  Strategy: Keep highest quality recordings first")
    
    balanced_dfs = []
    
    for species in tqdm(valid_species, desc="  Balancing species"):
        species_df = df_filtered[df_filtered['categories'] == species].copy()
        
        # Sort by quality (highest first), then by duration (longest first for efficiency)
        species_df = species_df.sort_values(
            by=['quality_score', 'duration'],
            ascending=[False, False]
        )
        
        # Greedily select recordings until we reach target duration
        cumsum_duration = 0
        selected_indices = []
        
        for idx, row in species_df.iterrows():
            if cumsum_duration >= target_duration:
                break
            
            selected_indices.append(idx)
            cumsum_duration += row['duration']
        
        selected_df = species_df.loc[selected_indices]
        balanced_dfs.append(selected_df)
        
        # Statistics
        actual_duration = selected_df['duration'].sum()
        n_recordings = len(selected_df)
        quality_dist = selected_df['q'].value_counts().to_dict()
        
        if len(balanced_dfs) <= 5:  # Print first 5 species as examples
            print(f"\n    {species}:")
            print(f"      Recordings: {n_recordings}")
            print(f"      Duration:   {actual_duration/60:.2f} min (target: {target_duration/60:.2f})")
            print(f"      Quality:    {quality_dist}")
    
    # Combine all balanced species
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"\n  ✓ Balanced dataset created")
    print(f"    Total recordings: {len(df_balanced)}")
    print(f"    Total species:    {df_balanced['categories'].nunique()}")
    print(f"    Total duration:   {df_balanced['duration'].sum()/3600:.2f} hours")
    
    # ==========================================
    # STEP 6: QUALITY DISTRIBUTION
    # ==========================================
    print(f"\n Quality distribution in balanced dataset:")
    quality_dist = df_balanced['q'].value_counts().sort_index()
    for quality, count in quality_dist.items():
        pct = 100.0 * count / len(df_balanced)
        print(f"    {quality}: {count:5d} ({pct:5.2f}%)")
    
    # ==========================================
    # STEP 7: VERIFY BALANCE
    # ==========================================
    print(f"\n Verifying balance...")
    
    final_species_counts = df_balanced['categories'].value_counts()
    final_species_durations = df_balanced.groupby('categories')['duration'].sum()
    
    print(f"\n  Recordings per species:")
    print(f"    Min:    {final_species_counts.min()}")
    print(f"    Max:    {final_species_counts.max()}")
    print(f"    Mean:   {final_species_counts.mean():.1f}")
    print(f"    Std:    {final_species_counts.std():.1f}")
    
    print(f"\n  Duration per species (minutes):")
    print(f"    Min:    {final_species_durations.min()/60:.2f}")
    print(f"    Max:    {final_species_durations.max()/60:.2f}")
    print(f"    Mean:   {final_species_durations.mean()/60:.2f}")
    print(f"    Std:    {final_species_durations.std()/60:.2f}")

    # Check if we have enough samples for train/val split
    min_needed = int(CONFIG["min_train_samples_per_species"] / CONFIG["train_ratio"])
    if final_species_counts.min() < min_needed:
        print(f"\n   WARNING: Some species have < {min_needed} recordings")
        print(f"     With {CONFIG['train_ratio']:.0%} train split, they'll have < {CONFIG['min_train_samples_per_species']} train samples")
        print(f"     This may cause issues with stratified splitting")
        print(f"\n  Species with few samples:")
        low_count_species = final_species_counts[final_species_counts < min_needed]
        for species, count in low_count_species.items():
            print(f"     {species}: {count} recordings")

    
    # ==========================================
    # STEP 8: SPLIT INTO TRAIN/TEST
    # ==========================================
    print(f"\n Splitting into train/test ({CONFIG['train_ratio']:.0%}/{1-CONFIG['train_ratio']:.0%})...")
    
    # Split per species to maintain balance
    train_dfs = []
    test_dfs = []
    
    np.random.seed(CONFIG['random_seed'])

    # Check minimum samples per species
    species_counts = df_balanced['categories'].value_counts()
    min_samples = species_counts.min()

    print(f"  Samples per species: Min={min_samples}, Max={species_counts.max()}, Mean={species_counts.mean():.1f}")

     # Calculate minimum needed for stratified split
    min_needed_for_split = max(2, int(np.ceil(1 / (1 - CONFIG['train_ratio']))))
    
    if min_samples < min_needed_for_split:
        print(f"\n   WARNING: Some species have < {min_needed_for_split} samples!")
        print(f"     This means they cannot be split {CONFIG['train_ratio']:.0%}/{1-CONFIG['train_ratio']:.0%}")
        print(f"     (minimum {min_needed_for_split} samples needed for a proper split)")
        
        # Show problematic species
        low_count_species = species_counts[species_counts < min_needed_for_split]
        print(f"\n  Problematic species ({len(low_count_species)}):")
        for species, count in low_count_species.items():
            print(f"    {species}: {count} samples")
        
        # Remove species with too few samples
        species_to_keep = species_counts[species_counts >= min_needed_for_split].index
        df_balanced = df_balanced[df_balanced['categories'].isin(species_to_keep)].copy()
        print(f"\n  ✓ Removed {len(low_count_species)} species")
        print(f"  ✓ Remaining: {len(df_balanced)} recordings from {df_balanced['categories'].nunique()} species")

    for species in df_balanced['categories'].unique():
        species_df = df_balanced[df_balanced['categories'] == species].copy()
        
        # Shuffle
        species_df = species_df.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
        
        # Split
        n_train = int(len(species_df) * CONFIG['train_ratio'])
        
        train_dfs.append(species_df.iloc[:n_train])
        test_dfs.append(species_df.iloc[n_train:])
    
    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test = pd.concat(test_dfs, ignore_index=True)
    
    # Shuffle again
    df_train = df_train.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
    
    print(f"  ✓ Train: {len(df_train)} recordings ({len(df_train['categories'].unique())} species)")
    print(f"  ✓ Test:  {len(df_test)} recordings ({len(df_test['categories'].unique())} species)")
    
    print(f"\n  Train duration: {df_train['duration'].sum()/3600:.2f} hours")
    print(f"  Test duration:  {df_test['duration'].sum()/3600:.2f} hours")
    
    # Verify all test species are in train
    train_species = set(df_train['categories'].unique())
    test_species = set(df_test['categories'].unique())
    missing_from_train = test_species - train_species
    
    if missing_from_train:
        print(f"\n   ERROR: {len(missing_from_train)} species in test but not train!")
        for sp in list(missing_from_train)[:5]:
            print(f"    - {sp}")
    
    print(f"\n  Train duration: {df_train['duration'].sum()/3600:.2f} hours")
    print(f"  Test duration:  {df_test['duration'].sum()/3600:.2f} hours")
    # ==========================================
    # STEP 9: SAVE DATASETS
    # ==========================================
    print(f"\n Saving datasets to {CONFIG['output_dir']}...")
    
    # Save full balanced dataset
    output_full = os.path.join(CONFIG['output_dir'], 'balanced_full.csv')
    df_balanced.to_csv(output_full, index=False, sep=';')
    print(f"  ✓ Saved full balanced dataset: {output_full}")
    
    # Save train split
    output_train = os.path.join(CONFIG['output_dir'], 'balanced_train.csv')
    df_train.to_csv(output_train, index=False, sep=';')
    print(f"  ✓ Saved train split: {output_train}")
    
    # Save test split
    output_test = os.path.join(CONFIG['output_dir'], 'balanced_test.csv')
    df_test.to_csv(output_test, index=False, sep=';')
    print(f"  ✓ Saved test split: {output_test}")
    
    # ==========================================
    # STEP 10: CREATE SUMMARY REPORT
    # ==========================================
    print(f"\n Creating summary report...")
    
    summary = {
        'dataset_info': {
            'total_recordings': len(df_balanced),
            'total_species': df_balanced['categories'].nunique(),
            'total_duration_hours': df_balanced['duration'].sum() / 3600,
            'target_duration_per_species_minutes': target_duration / 60,
        },
        'train_split': {
            'recordings': len(df_train),
            'species': df_train['categories'].nunique(),
            'duration_hours': df_train['duration'].sum() / 3600,
        },
        'test_split': {
            'recordings': len(df_test),
            'species': df_test['categories'].nunique(),
            'duration_hours': df_test['duration'].sum() / 3600,
        },
        'quality_distribution': quality_dist.to_dict(),
        'recordings_per_species': {
            'min': int(final_species_counts.min()),
            'max': int(final_species_counts.max()),
            'mean': float(final_species_counts.mean()),
            'std': float(final_species_counts.std()),
        },
        'duration_per_species_minutes': {
            'min': float(final_species_durations.min() / 60),
            'max': float(final_species_durations.max() / 60),
            'mean': float(final_species_durations.mean() / 60),
            'std': float(final_species_durations.std() / 60),
        },
        'config': CONFIG,
    }
    
    import json
    summary_path = os.path.join(CONFIG['output_dir'], 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved summary: {summary_path}")
    
    # Create per-species summary
    species_summary = []
    for species in df_balanced['categories'].unique():
        species_df = df_balanced[df_balanced['categories'] == species]
        species_summary.append({
            'species': species,
            'n_recordings': len(species_df),
            'total_duration_minutes': species_df['duration'].sum() / 60,
            'quality_A': len(species_df[species_df['q'] == 'A']),
            'quality_B': len(species_df[species_df['q'] == 'B']),
            'quality_C': len(species_df[species_df['q'] == 'C']),
            'quality_D': len(species_df[species_df['q'] == 'D']),
            'quality_E': len(species_df[species_df['q'] == 'E']),
        })
    
    species_summary_df = pd.DataFrame(species_summary)
    species_summary_df = species_summary_df.sort_values('species')
    
    species_summary_path = os.path.join(CONFIG['output_dir'], 'species_summary.csv')
    species_summary_df.to_csv(species_summary_path, index=False)
    print(f"  ✓ Saved per-species summary: {species_summary_path}")
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print(f"\n{'='*70}")
    print(f"DATASET BALANCING COMPLETE")
    print(f"{'='*70}")
    print(f"\n Final Statistics:")
    print(f"  Species:              {df_balanced['categories'].nunique()}")
    print(f"  Total Recordings:     {len(df_balanced)}")
    print(f"  Total Duration:       {df_balanced['duration'].sum()/3600:.2f} hours")
    print(f"\n  Train Set:            {len(df_train)} recordings ({df_train['duration'].sum()/3600:.2f} hours)")
    print(f"  Test Set:             {len(df_test)} recordings ({df_test['duration'].sum()/3600:.2f} hours)")
    print(f"\n  Target per species:   {target_duration/60:.2f} minutes")
    print(f"  Recordings/species:   {final_species_counts.min()}-{final_species_counts.max()} (μ={final_species_counts.mean():.1f})")
    print(f"  Duration/species:     {final_species_durations.min()/60:.2f}-{final_species_durations.max()/60:.2f} min")
    print(f"\n📁 Output Directory:    {CONFIG['output_dir']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()