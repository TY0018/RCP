import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "input_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset/balanced_test.csv",  # or balanced_full.csv
    "output_dir": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/expanded_test_dataset",
    "output_csv": "expanded_test.csv",
    "segment_length": 5.0,  # seconds
    "overlap": 0.0,  # seconds of overlap between segments (0 = no overlap)
    "sample_rate": 32000,
    "save_audio_segments": False,  # Set to True if you want to save actual audio files
    "audio_output_dir": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/expanded_audio",  # Only used if save_audio_segments=True
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
if CONFIG["save_audio_segments"]:
    os.makedirs(CONFIG["audio_output_dir"], exist_ok=True)

# ==========================================
# MAIN PROCESSING
# ==========================================
def parse_duration(length_str):
    """
    Parse duration from length column format.
    
    Args:
        length_str: String like "1:29" (1 min 29 sec) or "0:45" (45 sec)
    
    Returns:
        Duration in seconds (float)
    """
    try:
        if pd.isna(length_str):
            return None
        
        length_str = str(length_str).strip()
        
        # Handle format like "1:29" (minutes:seconds)
        if ':' in length_str:
            parts = length_str.split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            elif len(parts) == 3:
                # Handle format like "0:1:29" (hours:minutes:seconds)
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        
        # Handle plain number (assume seconds)
        return float(length_str)
        
    except Exception as e:
        print(f" Error parsing duration '{length_str}': {e}")
        return None

def segment_audio(audio_path, segment_length, overlap, sr):
    """
    Split audio into multiple segments.
    
    Returns:
        List of (start_time, end_time, duration) tuples
    """
    try:
        # Get duration without loading full audio (faster)
        duration = librosa.get_duration(path=audio_path, sr=sr)
        
        segments = []
        segment_samples = int(segment_length * sr)
        overlap_samples = int(overlap * sr)
        stride_samples = segment_samples - overlap_samples
        
        # Calculate all segment windows
        start = 0
        while start + segment_length <= duration:
            end = start + segment_length
            segments.append((start, end, segment_length))
            start += (segment_length - overlap)
        
        # Handle remainder if it's long enough (at least 80% of segment_length)
        if duration - start >= segment_length * 0.8:
            segments.append((start, duration, duration - start))
        
        return segments, duration
        
    except Exception as e:
        print(f" Error processing {audio_path}: {e}")
        return [], 0.0


def save_audio_segment(audio_path, start_time, end_time, output_path, sr):
    """
    Load and save a specific segment of audio.
    """
    try:
        # Load only the segment we need (more efficient)
        y, _ = librosa.load(audio_path, sr=sr, offset=start_time, duration=end_time-start_time, mono=True)
        
        # Save the segment
        sf.write(output_path, y, sr)
        
        return True
    except Exception as e:
        print(f" Error saving segment: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("EXPANDING DATASET BY SEGMENTING LONG RECORDINGS")
    print("="*70 + "\n")
    
    # ==========================================
    # STEP 1: LOAD ORIGINAL DATASET
    # ==========================================
    print(f" Loading dataset from {CONFIG['input_csv']}...")
    df_original = pd.read_csv(CONFIG["input_csv"], delimiter=';')
    
    print(f" Loaded {len(df_original)} recordings")
    print(f" Species: {df_original['categories'].nunique()}")
    
    # ==========================================
    # STEP 2: ANALYZE RECORDINGS
    # ==========================================
    print(f"\n Analyzing recordings...")
    
    # Get durations if not already in the dataframe
    has_length_column = 'length' in df_original.columns
    if 'duration' in df_original.columns:
        print(f"  Using existing 'duration' column (in seconds)")
        durations = df_original['duration'].values
    elif has_length_column:
        print(f"  Parsing 'length' column (format: M:SS)...")
        durations = []
        failed_parses = 0
        for idx, row in tqdm(df_original.iterrows(), total=len(df_original), desc="  Parsing"):
            duration = parse_duration(row['length'])
            if duration is None:
                # Fallback: compute from file
                try:
                    duration = librosa.get_duration(path=row['fullfilename'], sr=CONFIG["sample_rate"])
                except:
                    duration = 0.0
                    failed_parses += 1
            durations.append(duration)
        
        if failed_parses > 0:
            print(f"Failed to parse {failed_parses} durations, computed from audio files")
        
        df_original['duration'] = durations
    else:
        print(f"Computing durations from audio files...")
        durations = []
        for idx, row in tqdm(df_original.iterrows(), total=len(df_original), desc="  Processing"):
            filepath = row['fullfilename']
            try:
                duration = librosa.get_duration(path=filepath, sr=CONFIG["sample_rate"])
                durations.append(duration)
            except:
                durations.append(0.0)
        df_original['duration'] = durations
    
    
    # Statistics
    total_duration = df_original['duration'].sum()
    short_recs = len(df_original[df_original['duration'] <= CONFIG["segment_length"]])
    long_recs = len(df_original[df_original['duration'] > CONFIG["segment_length"]])
    
    print(f"\n  Duration statistics:")
    print(f"    Total duration:       {total_duration/3600:.2f} hours")
    print(f"    Recordings ≤ {CONFIG['segment_length']}s:  {short_recs} ({short_recs/len(df_original)*100:.1f}%)")
    print(f"    Recordings > {CONFIG['segment_length']}s:   {long_recs} ({long_recs/len(df_original)*100:.1f}%)")
    print(f"    Mean duration:        {df_original['duration'].mean():.2f}s")
    print(f"    Max duration:         {df_original['duration'].max():.2f}s")
    
    # ==========================================
    # STEP 3: SEGMENT LONG RECORDINGS
    # ==========================================
    print(f"\n Segmenting recordings longer than {CONFIG['segment_length']}s...")
    print(f"  Segment length: {CONFIG['segment_length']}s")
    print(f"  Overlap:        {CONFIG['overlap']}s")
    
    expanded_rows = []
    total_segments = 0
    
    for idx, row in tqdm(df_original.iterrows(), total=len(df_original), desc="  Processing"):
        filepath = row['fullfilename']
        duration = row['duration']
        
        if duration <= CONFIG["segment_length"]:
            # Keep short recordings as-is
            new_row = row.copy()
            new_row['segment_id'] = 0
            new_row['start_time'] = 0.0
            new_row['end_time'] = duration
            new_row['segment_duration'] = duration
            new_row['original_duration'] = duration
            new_row['is_segmented'] = False
            expanded_rows.append(new_row)
        else:
            # Segment long recordings
            segments, full_duration = segment_audio(
                filepath, 
                CONFIG["segment_length"], 
                CONFIG["overlap"],
                CONFIG["sample_rate"]
            )
            
            if len(segments) == 0:
                # Failed to segment, keep original
                new_row = row.copy()
                new_row['segment_id'] = 0
                new_row['start_time'] = 0.0
                new_row['end_time'] = duration
                new_row['segment_duration'] = duration
                new_row['original_duration'] = duration
                new_row['is_segmented'] = False
                expanded_rows.append(new_row)
                continue
            
            # Create entry for each segment
            for seg_id, (start, end, seg_dur) in enumerate(segments):
                new_row = row.copy()
                new_row['segment_id'] = seg_id
                new_row['start_time'] = start
                new_row['end_time'] = end
                new_row['segment_duration'] = seg_dur
                new_row['original_duration'] = full_duration
                new_row['is_segmented'] = True
                
                # Optionally save the audio segment as a new file
                if CONFIG["save_audio_segments"]:
                    # Create new filename
                    base_name = os.path.splitext(os.path.basename(filepath))[0]
                    ext = os.path.splitext(filepath)[1]
                    new_filename = f"{base_name}_seg{seg_id:03d}{ext}"
                    new_filepath = os.path.join(CONFIG["audio_output_dir"], new_filename)
                    
                    # Save segment
                    if save_audio_segment(filepath, start, end, new_filepath, CONFIG["sample_rate"]):
                        new_row['fullfilename'] = new_filepath
                
                expanded_rows.append(new_row)
                total_segments += 1
    
    # Create expanded dataframe
    df_expanded = pd.DataFrame(expanded_rows)
    
    print(f"\n  ✓ Original recordings: {len(df_original)}")
    print(f"  ✓ Expanded to:         {len(df_expanded)} segments")
    print(f"  ✓ New segments:        {total_segments}")
    print(f"  ✓ Expansion factor:    {len(df_expanded)/len(df_original):.2f}x")
    
    # ==========================================
    # STEP 4: STATISTICS BY SPECIES
    # ==========================================
    print(f"\n Per-species statistics:")
    
    species_stats = []
    for species in df_expanded['categories'].unique():
        orig_count = len(df_original[df_original['categories'] == species])
        exp_count = len(df_expanded[df_expanded['categories'] == species])
        expansion = exp_count / orig_count if orig_count > 0 else 0
        
        species_stats.append({
            'species': species,
            'original_count': orig_count,
            'expanded_count': exp_count,
            'expansion_factor': expansion
        })
    
    species_stats_df = pd.DataFrame(species_stats)
    species_stats_df = species_stats_df.sort_values('expansion_factor', ascending=False)
    
    print(f"\n  Top 10 species by expansion:")
    print(f"  {'Species':<30} {'Original':<10} {'Expanded':<10} {'Factor'}")
    print(f"  {'-'*65}")
    for _, row in species_stats_df.head(10).iterrows():
        print(f"  {row['species']:<30} {row['original_count']:<10} {row['expanded_count']:<10} {row['expansion_factor']:.2f}x")
    
    print(f"\n  Expansion statistics:")
    print(f"    Min:    {species_stats_df['expansion_factor'].min():.2f}x")
    print(f"    Max:    {species_stats_df['expansion_factor'].max():.2f}x")
    print(f"    Mean:   {species_stats_df['expansion_factor'].mean():.2f}x")
    print(f"    Median: {species_stats_df['expansion_factor'].median():.2f}x")
    
    # ==========================================
    # STEP 5: SAVE EXPANDED DATASET
    # ==========================================
    print(f"\n Saving expanded dataset...")
    
    output_path = os.path.join(CONFIG["output_dir"], CONFIG["output_csv"])
    df_expanded.to_csv(output_path, sep=';', index=False)
    print(f" Saved to: {output_path}")
    
    # Save species statistics
    stats_path = os.path.join(CONFIG["output_dir"], "expansion_stats.csv")
    species_stats_df.to_csv(stats_path, index=False)
    print(f" Stats saved to: {stats_path}")
    
    # ==========================================
    # STEP 6: SUMMARY
    # ==========================================
    print(f"\n{'='*70}")
    print(f"EXPANSION COMPLETE")
    print(f"{'='*70}")
    print(f"\n Summary:")
    print(f"  Original recordings:   {len(df_original)}")
    print(f"  Expanded segments:     {len(df_expanded)}")
    print(f"  New segments created:  {total_segments}")
    print(f"  Overall expansion:     {len(df_expanded)/len(df_original):.2f}x")
    print(f"\n  Original duration:     {df_original['duration'].sum()/3600:.2f} hours")
    print(f"  Expanded duration:     {df_expanded['segment_duration'].sum()/3600:.2f} hours")
    
    if CONFIG["save_audio_segments"]:
        print(f"\n  Audio segments saved to: {CONFIG['audio_output_dir']}")
    else:
        print(f"\n  Note: Audio segments NOT saved (save_audio_segments=False)")
        print(f"        The CSV contains start/end times for on-the-fly loading")
    
    print(f"\n Output: {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()