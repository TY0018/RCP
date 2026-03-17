"""
AudioProtoPNet Inference on Custom Asian Bird Dataset

Evaluates the pretrained AudioProtoPNet model on a custom test CSV,
computing AUROC, cmAP, and Top-1 Accuracy.

The script handles mapping between dataset categories (Genus_species format)
and AudioProtoPNet's ebird code labels.

Requirements:
    pip install transformers librosa pandas scikit-learn tqdm torch

Usage:
    python audioprotopnet_asian_inference.py
"""

import os
import time
import numpy as np
import pandas as pd
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import json

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Model
    "model_name": "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
    
    # Data - your custom test CSV (semicolon-delimited, same format as balanced_test.csv)
    "test_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_test.csv",
    
    # Bidirectional mapping: ebird_code <-> "Genus species_Common Name"
    "label2name_json": "/home/users/ntu/ytong005/dataset_json/label2name.json",
    
    # Parameters
    "sample_rate": 32000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ==========================================
# LABEL MAPPING
# ==========================================
def build_label_mapping(test_csv, model, label2name_path):
    """
    Build mapping: dataset category (Genus_species) -> model index.
    
    Flow:
      CSV category: Genus_species (e.g., Struthio_camelus)
      -> scientific name: "struthio camelus"
      -> find key in label2name.json that starts with "Struthio camelus_..."
      -> get ebird code (e.g., "ostric2")
      -> model.config.label2id["ostric2"] -> model index
    """
    df = pd.read_csv(test_csv, delimiter=';')
    dataset_categories = sorted(df['categories'].unique().tolist())
    
    model_label2id = model.config.label2id  # ebird_code -> model_index
    num_model_classes = len(model_label2id)
    
    # Load the bidirectional mapping
    with open(label2name_path, 'r') as f:
        label2name = json.load(f)
    
    # Build lookup: scientific_name_lower -> ebird_code
    # From entries like "Struthio camelus_Common Ostrich": "ostric2"
    scientific_to_ebird = {}
    for key, value in label2name.items():
        # If key contains "_" and value looks like an ebird code (short, no spaces)
        if '_' in key and ' ' not in value:
            # key = "Struthio camelus_Common Ostrich", value = "ostric2"
            scientific_name = key.split('_')[0].strip().lower()
            scientific_to_ebird[scientific_name] = value
    
    print(f"\n📊 Model has {num_model_classes} classes")
    print(f"📊 Dataset has {len(dataset_categories)} unique categories")
    print(f"📊 Taxonomy has {len(scientific_to_ebird)} scientific name -> ebird mappings")
    
    # Map each dataset category to a model index
    category_to_model_idx = {}
    unmapped = []
    
    for cat in dataset_categories:
        # Convert Genus_species -> "genus species" for lookup
        scientific = cat.replace('_', ' ').lower()
        
        if scientific in scientific_to_ebird:
            ebird_code = scientific_to_ebird[scientific]
            if ebird_code in model_label2id:
                category_to_model_idx[cat] = model_label2id[ebird_code]
            else:
                unmapped.append((cat, ebird_code, "ebird code not in model"))
        else:
            unmapped.append((cat, None, "scientific name not in taxonomy"))
    
    valid_indices = sorted(set(category_to_model_idx.values()))
    
    print(f"\n  ✓ Mapped {len(category_to_model_idx)}/{len(dataset_categories)} categories")
    print(f"  ✓ Valid model class indices: {len(valid_indices)}")
    
    if unmapped:
        print(f"\n  ⚠️  Unmapped categories ({len(unmapped)}):")
        for cat, ebird, reason in unmapped[:20]:
            print(f"      - {cat} ({reason})")
        if len(unmapped) > 20:
            print(f"      ... and {len(unmapped) - 20} more")
    
    return category_to_model_idx, valid_indices


# ==========================================
# INFERENCE
# ==========================================
def run_inference(test_csv, model, feature_extractor, category_to_model_idx, 
                  valid_indices, device):
    """
    Run AudioProtoPNet inference on each sample in the test CSV.
    """
    df = pd.read_csv(test_csv, delimiter=';')
    has_segments = 'start_time' in df.columns and 'end_time' in df.columns
    
    num_model_classes = len(model.config.id2label)
    sample_rate = CONFIG["sample_rate"]
    target_len = sample_rate * 5  # 5 seconds
    
    all_probs = []
    all_targets = []
    skipped = 0
    
    print(f"\n🔊 Running inference on {len(df)} samples...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="AudioProtoPNet Inference"):
        category = row['categories']
        audio_path = row['fullfilename']
        
        # Skip if category not mapped
        if category not in category_to_model_idx:
            skipped += 1
            continue
        
        true_model_idx = category_to_model_idx[category]
        
        # Build target vector
        target = np.zeros(num_model_classes)
        target[true_model_idx] = 1
        
        # Load audio
        try:
            if has_segments and pd.notna(row.get('start_time')) and pd.notna(row.get('end_time')):
                offset = float(row['start_time'])
                duration = float(row.get('segment_duration', row['end_time'] - offset))
                y, _ = librosa.load(audio_path, sr=sample_rate, mono=True,
                                    offset=offset, duration=duration)
            else:
                y, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        except Exception as e:
            print(f"\n❌ Audio loading error for {audio_path}: {e}")
            skipped += 1
            continue
        
        if len(y) == 0:
            skipped += 1
            continue
        
        # Process audio chunks (sliding window for longer clips)
        chunks = []
        if len(y) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:len(y)] = y
            chunks.append(padded)
        else:
            stride = int(target_len * 0.5)
            for start in range(0, len(y) - target_len + 1, stride):
                chunks.append(y[start:start + target_len])
            if len(chunks) == 0:
                chunks.append(y[-target_len:])
        
        # Run inference on each chunk
        chunk_probs = []
        with torch.no_grad():
            for chunk in chunks:
                inputs = feature_extractor([chunk], padding=True, return_tensors="pt")
                inputs = inputs.to(device)
                
                output = model(inputs)
                logits = output.logits
                
                # Mask logits to valid classes only
                mask = torch.full_like(logits, float('-inf'))
                mask[:, valid_indices] = logits[:, valid_indices]
                
                # Sigmoid for multi-label probabilities (matching audioprotopnet_inference.py)
                probs = torch.sigmoid(mask).cpu().numpy().squeeze()
                chunk_probs.append(probs)
        
        # Max-pool across chunks
        if len(chunk_probs) > 1:
            final_probs = np.max(np.array(chunk_probs), axis=0)
        else:
            final_probs = chunk_probs[0]
        
        all_probs.append(final_probs)
        all_targets.append(target)
    
    if skipped > 0:
        print(f"\n⚠️  Skipped {skipped} samples (unmapped categories or audio errors)")
    
    return np.array(all_probs), np.array(all_targets)


# ==========================================
# METRICS
# ==========================================
def compute_metrics(all_probs, all_targets, valid_indices):
    """
    Compute AUROC, cmAP, and Top-1 Accuracy (matching audioprotopnet_inference.py).
    """
    print("\n📊 Computing metrics...")
    
    # cmAP = mean of per-class AP (only over valid classes with positive samples)
    cmAP = np.mean([
        average_precision_score(all_targets[:, i], all_probs[:, i])
        for i in valid_indices
        if np.sum(all_targets[:, i]) > 0
    ])
    
    # AUROC = mean per-class ROC AUC
    try:
        AUROC = np.mean([
            roc_auc_score(all_targets[:, i], all_probs[:, i])
            for i in valid_indices
            if len(np.unique(all_targets[:, i])) > 1
        ])
    except ValueError:
        AUROC = float("nan")
    
    # Top-1 Accuracy (among valid classes only)
    pred_indices = np.argmax(all_probs[:, valid_indices], axis=1)
    pred_labels = [valid_indices[idx] for idx in pred_indices]
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
# BENCHMARKING
# ==========================================
def benchmark_model(model, feature_extractor, device, num_runs=50):
    """Measure model size, inference time, GPU memory, and FLOPs."""
    print("\n" + "=" * 60)
    print("MODEL BENCHMARK")
    print("=" * 60)
    
    # --- Parameter count & model size ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)  # FP32
    
    print(f"\n  Total params:       {total_params:,}")
    print(f"  Trainable params:   {trainable_params:,}")
    print(f"  Model size (FP32):  {model_size_mb:.1f} MB")
    
    # --- Create dummy input (5s audio at 32kHz) ---
    dummy_audio = np.random.randn(CONFIG["sample_rate"] * 5).astype(np.float32)
    dummy_input = feature_extractor([dummy_audio], padding=True, return_tensors="pt")
    dummy_input = dummy_input.to(device)
    
    # --- GPU Memory (if CUDA) ---
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"  Peak GPU memory:    {peak_mem:.1f} MB")
    
    # --- Inference latency ---
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    avg_time = (time.perf_counter() - start) / num_runs
    
    print(f"  Avg inference time: {avg_time * 1000:.2f} ms")
    print(f"  Throughput:         {1 / avg_time:.1f} samples/sec")
    
    # --- FLOPs (optional, requires thop) ---
    try:
        from thop import profile as thop_profile
        flops, _ = thop_profile(model, inputs=(dummy_input,), verbose=False)
        print(f"  FLOPs:              {flops / 1e9:.2f} GFLOPs")
    except ImportError:
        print(f"  FLOPs:              (install 'thop' to measure: pip install thop)")
    except Exception as e:
        print(f"  FLOPs:              (could not measure: {e})")
    
    print(f"{'=' * 60}\n")
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "avg_inference_ms": avg_time * 1000,
        "throughput_per_sec": 1 / avg_time,
    }


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AUDIOPROTOPNET INFERENCE ON CUSTOM DATASET")
    print("=" * 60 + "\n")
    
    device = torch.device(CONFIG["device"])
    
    # Load model
    print("🤖 Loading AudioProtoPNet model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"], trust_remote_code=True
    )
    model.eval().to(device)
    print(f"  ✓ Model loaded with {len(model.config.label2id)} classes")
    
    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        CONFIG["model_name"], trust_remote_code=True
    )
    
    # Build label mapping
    print(f"\n📂 Test CSV: {CONFIG['test_csv']}")
    category_to_model_idx, valid_indices = build_label_mapping(
        CONFIG["test_csv"], model, CONFIG["label2name_json"]
    )
    
    # Benchmark model
    benchmark_results = benchmark_model(model, feature_extractor, device)
    
    # Run inference (timed)
    print("\n🔊 Running inference...")
    inference_start = time.perf_counter()
    
    all_probs, all_targets = run_inference(
        CONFIG["test_csv"], model, feature_extractor, 
        category_to_model_idx, valid_indices, device
    )
    
    total_inference_time = time.perf_counter() - inference_start
    
    print(f"\n📊 Collected {len(all_probs)} predictions")
    print(f"⏱️  Total inference time: {total_inference_time:.1f}s ")
    print(f"⏱️  Avg per sample: {total_inference_time/max(len(all_probs),1)*1000:.1f}ms")
    
    # Compute and print metrics
    cmAP, AUROC, t1_acc = compute_metrics(all_probs, all_targets, valid_indices)
    
    print(f"\n{'=' * 60}")
    print(f"TEST RESULTS:")
    print(f"  cmAP:   {cmAP:.4f}")
    print(f"  AUROC:  {AUROC:.4f}")
    print(f"  T1-Acc: {t1_acc:.4f}")
    print(f"{'=' * 60}")
    print(f"\nBENCHMARK SUMMARY:")
    print(f"  Total params:       {benchmark_results['total_params']:,}")
    print(f"  Model size (FP32):  {benchmark_results['model_size_mb']:.1f} MB")
    print(f"  Avg inference time: {benchmark_results['avg_inference_ms']:.2f} ms")
    print(f"  Throughput:         {benchmark_results['throughput_per_sec']:.1f} samples/sec")
    print(f"  Total test time:    {total_inference_time:.1f}s ({len(all_probs)} samples)")
    print(f"{'=' * 60}\n")
