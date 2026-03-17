"""
BirdNET Inference on Custom Test Dataset

Evaluates the BirdNET model on a test CSV (same format as balanced_test.csv),
computing AUROC, cmAP, and Top-1 Accuracy.

"""

import os
import time
import tempfile
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# BirdNET v0.1.7 API
from birdnet import predict_species_within_audio_file, SpeciesPredictions
from birdnet import predict_species_at_location_and_time

# ==========================================
# 1. CONFIGURATION
# ==========================================

CONFIG = {
    # Paths
    "train_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_train.csv",
    "test_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_test.csv",
    
    # Parameters
    "sample_rate": 32000,
}

# ==========================================
# 1. CONFIGURATION
# ==========================================
TEST_CSV = CONFIG["test_csv"]
SAMPLE_RATE = CONFIG["sample_rate"]  # 32000


# ==========================================
# 2. GET BIRDNET SPECIES LIST
# ==========================================
def get_birdnet_species_list():
    """
    Get BirdNET's full species list from the acoustic model directly.
    The acoustic model has a .species attribute with all 6,522 labels.
    (predict_species_at_location_and_time only returns geographically-filtered species.)
    """
    from birdnet.models.v2m4.model_v2m4_protobuf import AudioModelV2M4Protobuf
    model = AudioModelV2M4Protobuf()
    all_species = list(model.species)
    return all_species


# ==========================================
# 3. BUILD LABEL MAPPING
# ==========================================
def build_birdnet_label_mapping(test_csv, all_birdnet_species):
    """
    Build mapping between dataset categories (Genus_species) and BirdNET species labels.
    
    Dataset categories: Genus_species (e.g. Falco_tinnunculus)
    BirdNET labels:     Genus species_Common Name (e.g. Falco tinnunculus_Common Kestrel)
    """
    df = pd.read_csv(test_csv, delimiter=';')
    dataset_categories = sorted(df['categories'].unique().tolist())
    
    # Build lookup: scientific_name_lower -> BirdNET label
    scientific_to_label = {}
    for label in all_birdnet_species:
        # BirdNET label format: "Genus species_Common Name"
        scientific_name = label.split("_")[0].strip().lower()
        scientific_to_label[scientific_name] = label
    
    species_to_idx = {sp: idx for idx, sp in enumerate(all_birdnet_species)}
    
    # Map each dataset category to a BirdNET label
    category_to_birdnet_label = {}
    unmapped = []
    
    for cat in dataset_categories:
        # Convert Genus_species -> genus species (lowercase for matching)
        scientific = cat.replace("_", " ").lower()
        if scientific in scientific_to_label:
            category_to_birdnet_label[cat] = scientific_to_label[scientific]
        else:
            unmapped.append(cat)
    
    valid_birdnet_indices = sorted([
        species_to_idx[label] 
        for label in category_to_birdnet_label.values()
    ])
    
    print(f"✓ Dataset has {len(dataset_categories)} unique categories")
    print(f"✓ Mapped {len(category_to_birdnet_label)}/{len(dataset_categories)} to BirdNET labels")
    if unmapped:
        print(f" Unmapped categories ({len(unmapped)}): {unmapped}")
    print(f"✓ Valid BirdNET class indices: {len(valid_birdnet_indices)}")
    
    return category_to_birdnet_label, valid_birdnet_indices, species_to_idx


# ==========================================
# 4. INFERENCE
# ==========================================
def run_inference(test_csv, category_to_birdnet_label, species_to_idx, num_species):
    """
    Run BirdNET inference on each sample in the test CSV.
    
    Uses birdnet v0.1.7 API:
        predict_species_within_audio_file(path) -> list of ((start, end), {species: conf})
    """
    df = pd.read_csv(test_csv, delimiter=';')
    has_segments = 'start_time' in df.columns and 'end_time' in df.columns
    
    all_probs = []
    all_targets = []
    skipped = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running BirdNET Inference"):
        category = row['categories']
        audio_path = row['fullfilename']
        
        if category not in category_to_birdnet_label:
            skipped += 1
            continue
        
        true_label = category_to_birdnet_label[category]
        true_idx = species_to_idx[true_label]
        
        target = np.zeros(num_species)
        target[true_idx] = 1
        
        # Load audio segment
        try:
            if has_segments and pd.notna(row.get('start_time')) and pd.notna(row.get('end_time')):
                offset = float(row['start_time'])
                duration = float(row['segment_duration']) if 'segment_duration' in row else float(row['end_time']) - offset
                y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True,
                                    offset=offset, duration=duration)
            else:
                y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"\n Audio loading error for {audio_path}: {e}")
            skipped += 1
            continue
        
        if len(y) == 0:
            skipped += 1
            continue
        
        # Save to temp WAV (BirdNET works with file paths)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, y, SAMPLE_RATE)
            
            # Run BirdNET prediction (v0.1.7 API)
            # Returns list of ((start, end), SpeciesPrediction({species: conf}))
            raw_predictions = predict_species_within_audio_file(Path(tmp_path))
            predictions = SpeciesPredictions(raw_predictions)
            
            os.unlink(tmp_path)
        except Exception as e:
            print(f"\n BirdNET prediction error for {audio_path}: {e}")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            skipped += 1
            continue
        
        # Build probability vector from predictions
        # predictions is dict-like: {(start, end): {species: confidence}}
        prob_vector = np.zeros(num_species)
        
        for time_interval, species_pred in predictions.items():
            for species_label, confidence in species_pred.items():
                if species_label in species_to_idx:
                    idx = species_to_idx[species_label]
                    prob_vector[idx] = max(prob_vector[idx], confidence)
        
        all_probs.append(prob_vector)
        all_targets.append(target)
    
    if skipped > 0:
        print(f"\n  Skipped {skipped} samples (unmapped categories or audio errors)")
    
    return np.array(all_probs), np.array(all_targets)


# ==========================================
# 5. METRICS
# ==========================================
def compute_metrics(all_probs, all_targets, valid_birdnet_indices):
    """
    Compute AUROC, cmAP, and Top-1 Accuracy (matching audioprotopnet_inference.py).
    """
    print("\nCalculate metrics")
    
    # cmAP = mean of per-class AP (only over valid classes with positive samples)
    cmAP = np.mean([
        average_precision_score(all_targets[:, i], all_probs[:, i])
        for i in valid_birdnet_indices
        if np.sum(all_targets[:, i]) > 0
    ])
    
    # AUROC = mean per-class ROC AUC
    try:
        AUROC = np.mean([
            roc_auc_score(all_targets[:, i], all_probs[:, i])
            for i in valid_birdnet_indices
            if len(np.unique(all_targets[:, i])) > 1
        ])
    except ValueError:
        AUROC = float("nan")
    
    # Top-1 Accuracy (among valid classes only)
    pred_indices = np.argmax(all_probs[:, valid_birdnet_indices], axis=1)
    pred_labels = [valid_birdnet_indices[idx] for idx in pred_indices]
    correct = []
    
    for i in range(len(all_targets)):
        true_labels = np.where(all_targets[i] == 1)[0]
        if len(true_labels) == 0:
            continue
        if pred_labels[i] in true_labels:
            correct.append(1)
        else:
            correct.append(0)
    
    t1_acc = np.mean(correct) if len(correct) > 0 else 0.0
    
    return cmAP, AUROC, t1_acc


# ==========================================
# 6. BENCHMARKING
# ==========================================

def benchmark_birdnet(num_runs=50):
    """
    Benchmark BirdNET model: params, size, inference latency, GPU memory.
    BirdNET uses TensorFlow internally, so we access the TF model directly.
    """
    from birdnet.models.v2m4.model_v2m4_protobuf import AudioModelV2M4Protobuf
    
    print("\n" + "=" * 60)
    print("BIRDNET MODEL BENCHMARK")
    print("=" * 60)
    
    model = AudioModelV2M4Protobuf()
    
    # --- Parameter count (TensorFlow) ---
    try:
        import tensorflow as tf
        
        # Access the underlying TF model
        if hasattr(model, '_model'):
            tf_model = model._model
        elif hasattr(model, 'model'):
            tf_model = model.model
        else:
            tf_model = None
        
        if tf_model is not None and hasattr(tf_model, 'count_params'):
            total_params = tf_model.count_params()
            model_size_mb = total_params * 4 / (1024 ** 2)
            print(f"\n  Total params:       {total_params:,}")
            print(f"  Model size (FP32):  {model_size_mb:.1f} MB")
        else:
            # Try counting from graph variables
            total_params = sum(
                np.prod(v.shape) for v in tf.compat.v1.global_variables()
            ) if hasattr(tf, 'compat') else 0
            model_size_mb = total_params * 4 / (1024 ** 2) if total_params > 0 else 0
            if total_params > 0:
                print(f"\n  Total params:       {total_params:,}")
                print(f"  Model size (FP32):  {model_size_mb:.1f} MB")
            else:
                print(f"\n  Total params:       (could not access TF graph)")
                model_size_mb = 0
                
        # Check if GPU is being used
        gpus = tf.config.list_physical_devices('GPU')
        print(f"  TF GPUs available:  {len(gpus)}")
        if gpus:
            for gpu in gpus:
                print(f"    - {gpu.name}")
                
    except Exception as e:
        print(f"\n  Parameter count:    (error: {e})")
        total_params = 0
        model_size_mb = 0
    
    # --- Inference latency ---
    # Create a dummy 5s audio file for benchmarking
    dummy_audio = np.random.randn(SAMPLE_RATE * 5).astype(np.float32)
    tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(tmp_path, dummy_audio, SAMPLE_RATE)
    
    # Warm-up
    print(f"\n  Running {num_runs} benchmark iterations (after 5 warm-up)...")
    for _ in range(5):
        _ = list(predict_species_within_audio_file(
            Path(tmp_path), min_confidence=0.0, silent=True
        ))
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = list(predict_species_within_audio_file(
            Path(tmp_path), min_confidence=0.0, silent=True
        ))
    avg_time = (time.perf_counter() - start) / num_runs
    
    os.unlink(tmp_path)
    
    print(f"  Avg inference time: {avg_time * 1000:.2f} ms")
    print(f"  Throughput:         {1 / avg_time:.1f} samples/sec")
    
    # --- GPU Memory (TensorFlow) ---
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            peak_mem = mem_info.get('peak', 0) / (1024 ** 2)
            print(f"  Peak GPU memory:    {peak_mem:.1f} MB")
    except Exception:
        pass
    
    print(f"{'=' * 60}\n")
    
    return {
        "total_params": total_params,
        "model_size_mb": model_size_mb,
        "avg_inference_ms": avg_time * 1000,
        "throughput_per_sec": 1 / avg_time,
    }


# ==========================================
# 7. MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BIRDNET INFERENCE ON CUSTOM TEST DATASET")
    print("=" * 60 + "\n")
    
    # Get full species list via the geo model
    print(" Retrieving BirdNET species list...")
    all_birdnet_species = get_birdnet_species_list()
    num_species = len(all_birdnet_species)
    print(f"✓ BirdNET has {num_species} species classes")
    
    # Build mapping
    print(f"\n Loading test CSV: {TEST_CSV}")
    category_to_birdnet_label, valid_birdnet_indices, species_to_idx = \
        build_birdnet_label_mapping(TEST_CSV, all_birdnet_species)
    
    # Benchmark model
    benchmark_results = benchmark_birdnet()
    
    # Run inference (timed)
    print(f"\n Running inference...")
    inference_start = time.perf_counter()
    
    all_probs, all_targets = run_inference(
        TEST_CSV, category_to_birdnet_label,
        species_to_idx, num_species
    )
    
    total_inference_time = time.perf_counter() - inference_start
    
    print(f"\n Collected {len(all_probs)} predictions")
    print(f"⏱  Total inference time: {total_inference_time:.1f}s")
    print(f"⏱  Avg per sample: {total_inference_time / max(len(all_probs), 1) * 1000:.1f}ms")
    
    # Compute and print metrics
    cmAP, AUROC, t1_acc = compute_metrics(all_probs, all_targets, valid_birdnet_indices)
    
    print(f"\n{'=' * 60}")
    print(f"TEST RESULTS:")
    print(f"  cmAP:   {cmAP:.4f}")
    print(f"  AUROC:  {AUROC:.4f}")
    print(f"  T1-Acc: {t1_acc:.4f}")
    print(f"{'=' * 60}")
    print(f"\nBENCHMARK SUMMARY:")
    if benchmark_results['total_params'] > 0:
        print(f"  Total params:       {benchmark_results['total_params']:,}")
        print(f"  Model size (FP32):  {benchmark_results['model_size_mb']:.1f} MB")
    print(f"  Avg inference time: {benchmark_results['avg_inference_ms']:.2f} ms")
    print(f"  Throughput:         {benchmark_results['throughput_per_sec']:.1f} samples/sec")
    print(f"  Total test time:    {total_inference_time:.1f}s ({len(all_probs)} samples)")
    print(f"{'=' * 60}\n")
