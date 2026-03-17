import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Input files
    "full_dataset_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/SG_Birds/xc_metadata.csv",           # Complete dataset with all species
    "known_species_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_test.csv",  # Training species
    
    # Output
    "output_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/open_test1.csv",
    
    # Open-set composition
    "n_known_samples": 2000,      # Samples from known species (will use existing segments)
    "n_unknown_samples": 2000,    # Samples from unknown species (will be segmented)
    
    # Segmentation for unknown species
    "segment_unknown": True,     # Segment unknown species before adding to test
    "segment_length": 5.0,       # seconds
    "segment_overlap": 0.0,      # seconds
    
    # Other settings
    "sample_rate": 32000,
    "random_seed": 42,
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def parse_duration(length_str):
    """Parse duration from length column format (M:SS)."""
    try:
        if pd.isna(length_str):
            return None
        
        length_str = str(length_str).strip()
        
        if ':' in length_str:
            parts = length_str.split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        
        return float(length_str)
        
    except Exception as e:
        return None

def segment_recording(row, segment_length, overlap):
    """
    Segment a single recording into multiple parts.
    
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
    print("CREATING OPEN-SET TEST DATASET")
    print("="*70 + "\n")
    
    # ==========================================
    # STEP 1: LOAD DATASETS
    # ==========================================
    print(" Loading datasets...")
    df_full = pd.read_csv(CONFIG["full_dataset_csv"], delimiter=';')
    df_known = pd.read_csv(CONFIG["known_species_csv"], delimiter=';')
    
    print(f"  Full dataset: {len(df_full)} recordings")
    print(f"  Known species dataset: {len(df_known)} recordings")
    
    # Get known species
    known_species = set(df_known['categories'].unique())
    print(f"\n  Known species (trained on): {len(known_species)}")
    
    # Get all species and identify unknown
    all_species = set(df_full['categories'].unique())
    unknown_species = all_species - known_species
    print(f"  Total species in full dataset: {len(all_species)}")
    print(f"  Unknown species: {len(unknown_species)}")
    
    if len(unknown_species) == 0:
        print("\n   WARNING: No unknown species found!")
        print("     Your open-set test will only contain known species.")
        return
    
    print(f"\n  Example unknown species: {sorted(list(unknown_species))[:5]}")
    
    # ==========================================
    # STEP 2: SAMPLE KNOWN SPECIES
    # ==========================================
    print(f"\n Step 1: Sampling KNOWN species...")
    print("-" * 70)
    
    # Check if known dataset has segmentation columns
    has_segments_known = all(col in df_known.columns 
                            for col in ['start_time', 'end_time', 'segment_duration'])
    
    if has_segments_known:
        print(f"  ✓ Known species dataset already has segmentation")
    
    # Sample from known species
    if len(df_known) >= CONFIG["n_known_samples"]:
        df_known_test = df_known.sample(
            n=CONFIG["n_known_samples"],
            random_state=CONFIG["random_seed"]
        )
    else:
        print(f" Only {len(df_known)} known samples available (requested {CONFIG['n_known_samples']})")
        df_known_test = df_known.copy()
    
    print(f"  Sampled {len(df_known_test)} known species samples")
    
    # ==========================================
    # STEP 3: SAMPLE UNKNOWN SPECIES
    # ==========================================
    print(f"\n Step 2: Sampling UNKNOWN species...")
    print("-" * 70)
    
    df_unknown = df_full[df_full['categories'].isin(unknown_species)]
    print(f"  Unknown recordings available: {len(df_unknown)}")
    
    # Sample unknown recordings
    if len(df_unknown) >= CONFIG["n_unknown_samples"]:
        df_unknown_sampled = df_unknown.sample(
            n=CONFIG["n_unknown_samples"],
            random_state=CONFIG["random_seed"]
        )
    else:
        print(f" Only {len(df_unknown)} unknown samples available")
        df_unknown_sampled = df_unknown.copy()
    
    print(f"  Sampled {len(df_unknown_sampled)} unknown species samples")
    
    # ==========================================
    # STEP 4: SEGMENT UNKNOWN SPECIES
    # ==========================================
    if CONFIG["segment_unknown"]:
        print(f"\n Step 3: Segmenting UNKNOWN species...")
        print("-" * 70)
        print(f"  Segment length: {CONFIG['segment_length']}s")
        print(f"  Overlap: {CONFIG['segment_overlap']}s")
        
        # Get durations for unknown samples
        print(f"\n  Computing durations...")
        durations = []
        has_length_col = 'length' in df_unknown_sampled.columns
        
        if 'duration' in df_unknown_sampled.columns:
            print(f"    Using existing 'duration' column")
            durations = df_unknown_sampled['duration'].values
        elif has_length_col:
            print(f"    Parsing 'length' column...")
            for idx, row in tqdm(df_unknown_sampled.iterrows(), 
                                total=len(df_unknown_sampled), desc="    Parsing"):
                duration = parse_duration(row['length'])
                if duration is None:
                    try:
                        duration = librosa.get_duration(
                            path=row['fullfilename'], 
                            sr=CONFIG['sample_rate']
                        )
                    except:
                        duration = 0.0
                durations.append(duration)
        else:
            print(f"    Computing from audio files...")
            for idx, row in tqdm(df_unknown_sampled.iterrows(), 
                                total=len(df_unknown_sampled), desc="    Processing"):
                try:
                    duration = librosa.get_duration(
                        path=row['fullfilename'],
                        sr=CONFIG['sample_rate']
                    )
                except:
                    duration = 0.0
                durations.append(duration)
        
        df_unknown_sampled = df_unknown_sampled.copy()
        df_unknown_sampled['duration'] = durations
        
        # Remove recordings with 0 duration
        df_unknown_sampled = df_unknown_sampled[df_unknown_sampled['duration'] > 0].copy()
        
        print(f"\n  Duration statistics:")
        print(f"    Min:  {df_unknown_sampled['duration'].min():.2f}s")
        print(f"    Max:  {df_unknown_sampled['duration'].max():.2f}s")
        print(f"    Mean: {df_unknown_sampled['duration'].mean():.2f}s")
        
        # Segment recordings
        print(f"\n  Segmenting recordings...")
        all_segments = []
        
        for idx, row in tqdm(df_unknown_sampled.iterrows(), 
                           total=len(df_unknown_sampled), desc="  Segmenting"):
            segments = segment_recording(
                row, 
                CONFIG['segment_length'], 
                CONFIG['segment_overlap']
            )
            all_segments.extend(segments)
        
        df_unknown_test = pd.DataFrame(all_segments).reset_index(drop=True)
        if len(df_unknown_test) > CONFIG["n_unknown_samples"]:
            print(f"\n   Balancing: Reducing {len(df_unknown_test)} segments to 2000...")
            df_unknown_test = df_unknown_test.sample(
                n=CONFIG["n_unknown_samples"], 
                random_state=CONFIG['random_seed']
            ).reset_index(drop=True)
        
        print(f"\n  ✓ Segmentation complete:")
        print(f"    Original recordings: {len(df_unknown_sampled)}")
        print(f"    After segmentation:  {len(df_unknown_test)}")
        print(f"    Expansion factor:    {len(df_unknown_test)/len(df_unknown_sampled):.2f}x")
    else:
        df_unknown_test = df_unknown_sampled.copy()
        print(f"\n  Skipping segmentation (segment_unknown=False)")
    
    # ==========================================
    # STEP 5: COMBINE AND ALIGN COLUMNS
    # ==========================================
    print(f"\n Step 4: Combining known and unknown samples...")
    print("-" * 70)
    
    # Add segmentation columns to known if missing
    if not has_segments_known:
        print(f"  Adding segmentation columns to known samples...")
        df_known_test['segment_id'] = 0
        df_known_test['start_time'] = 0.0
        
        # Compute end_time from duration or length
        if 'duration' not in df_known_test.columns:
            if 'length' in df_known_test.columns:
                df_known_test['duration'] = df_known_test['length'].apply(parse_duration)
            else:
                df_known_test['duration'] = 5.0  # Default
        
        df_known_test['end_time'] = df_known_test['duration']
        df_known_test['segment_duration'] = df_known_test['duration']
        df_known_test['original_duration'] = df_known_test['duration']
        df_known_test['is_segmented'] = False
    
    # Ensure both have the same columns
    known_cols = set(df_known_test.columns)
    unknown_cols = set(df_unknown_test.columns)
    
    # Add missing columns
    for col in unknown_cols - known_cols:
        df_known_test[col] = None
    
    for col in known_cols - unknown_cols:
        df_unknown_test[col] = None
    
    # Combine
    df_test = pd.concat([df_known_test, df_unknown_test], ignore_index=True)
    
    # Shuffle
    df_test = df_test.sample(frac=1, random_state=CONFIG['random_seed']).reset_index(drop=True)
    
    print(f"  ✓ Combined dataset: {len(df_test)} samples")
    
    # ==========================================
    # STEP 6: STATISTICS
    # ==========================================
    print(f"\n Step 5: Final statistics...")
    print("-" * 70)
    
    # Known vs unknown
    known_in_test = df_test[df_test['categories'].isin(known_species)]
    unknown_in_test = df_test[~df_test['categories'].isin(known_species)]
    
    print(f"\n  Composition:")
    print(f"    Known samples:   {len(known_in_test)} ({len(known_in_test)/len(df_test)*100:.1f}%)")
    print(f"    Unknown samples: {len(unknown_in_test)} ({len(unknown_in_test)/len(df_test)*100:.1f}%)")
    
    print(f"\n  Species:")
    print(f"    Known species:   {len(known_in_test['categories'].unique())}")
    print(f"    Unknown species: {len(unknown_in_test['categories'].unique())}")
    
    print(f"\n  Top 5 known species:")
    for species, count in known_in_test['categories'].value_counts().head(5).items():
        print(f"    {species}: {count} samples")
    
    print(f"\n  Top 5 unknown species:")
    for species, count in unknown_in_test['categories'].value_counts().head(5).items():
        print(f"    {species}: {count} samples")
    
    # ==========================================
    # STEP 7: SAVE
    # ==========================================
    print(f"\n Step 6: Saving...")
    print("-" * 70)
    
    df_test.to_csv(CONFIG["output_csv"], sep=';', index=False)
    print(f" Saved to: {CONFIG['output_csv']}")
    
    # Save metadata
    import json
    metadata = {
        'total_samples': len(df_test),
        'known_samples': len(known_in_test),
        'unknown_samples': len(unknown_in_test),
        'known_species_count': len(known_in_test['categories'].unique()),
        'unknown_species_count': len(unknown_in_test['categories'].unique()),
        'known_species': sorted(list(known_species)),
        'unknown_species': sorted(list(unknown_species)),
        'segmented_unknown': CONFIG['segment_unknown'],
        'segment_length': CONFIG['segment_length'],
        'config': CONFIG
    }
    
    metadata_file = CONFIG["output_csv"].replace('.csv', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f" Saved metadata to: {metadata_file}")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*70}")
    print(f"OPEN-SET TEST DATASET CREATED")
    print(f"{'='*70}")
    print(f"\n✓ Total samples: {len(df_test)}")
    print(f"✓ Known:   {len(known_in_test)} samples from {len(known_in_test['categories'].unique())} species")
    print(f"✓ Unknown: {len(unknown_in_test)} samples from {len(unknown_in_test['categories'].unique())} species")
    print(f"\n✓ Output: {CONFIG['output_csv']}")
    print(f"✓ Metadata: {metadata_file}")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()