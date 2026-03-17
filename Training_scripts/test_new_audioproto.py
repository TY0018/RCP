"""
Prototype Model Testing with Benchmarks

Standalone evaluation script for the prototype-based AudioProtoPNet classifier 
trained in new_classification.py. Computes classification metrics (Top-1, Top-3, 
cmAP, AUROC) and model benchmarks (params, memory, latency, throughput, FLOPs).

"""

import os
import time
import math
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Paths
    "train_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_train.csv",
    "test_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_test.csv",
    "base_model": "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
    "model_checkpoint": "/home/users/ntu/ytong005/RCP/trained_classifier_proto/best_new_model.pth",
    
    # Prototype parameters
    "protos_per_class": 5,
    
    # Parameters
    "sample_rate": 32000,
    "batch_size": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Output
    "output_dir": "../test_results_proto",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==========================================
# MODEL
# ==========================================

class MultiPrototypeLayer(nn.Module):
    def __init__(self, feature_dim, num_classes, protos_per_class=5):
        super().__init__()
        self.num_classes = num_classes
        self.protos_per_class = protos_per_class
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, protos_per_class, feature_dim))
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))

    def forward(self, x):
        x_expanded = x.unsqueeze(1).unsqueeze(1)
        p_expanded = self.prototypes.unsqueeze(0)
        distances = torch.sum((x_expanded - p_expanded) ** 2, dim=-1)
        min_distances, _ = torch.min(distances, dim=2)
        return -min_distances, min_distances


class AudioProtoPNetClassifier(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        
        print(f"Loading pretrained model: {base_model_name}")
        full_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        
        if hasattr(full_model, "backbone"):
            self.backbone = full_model.backbone
        elif hasattr(full_model, "base_model"):
            self.backbone = full_model.base_model
        else:
            self.backbone = full_model

        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.feature_dim = self._get_feature_dim()
        
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )
        
        self.proto_layer = MultiPrototypeLayer(
            512, num_classes, CONFIG["protos_per_class"]
        )
    
    def _get_feature_dim(self):
        try:
            if hasattr(self.backbone.config, "hidden_size"):
                return self.backbone.config.hidden_size
            if hasattr(self.backbone.config, "num_features"):
                return self.backbone.config.num_features
        except:
            pass
        return 1024

    def extract_features(self, inputs):
        with torch.no_grad():
            outputs = self.backbone(inputs, output_hidden_states=True)
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                features = outputs.hidden_states[-1]
            else:
                features = outputs

        if features.dim() == 3:
            features = features.mean(dim=1)
        elif features.dim() == 4:
            features = features.mean(dim=(2, 3))
                
        return self.projection(features)
    
    def forward(self, x):
        feats = self.extract_features(x)
        logits, min_distances = self.proto_layer(feats)
        return logits, min_distances


# ==========================================
# DATASET
# ==========================================

class TestDataset(Dataset):
    def __init__(self, csv_file, category2id):
        self.df = pd.read_csv(csv_file, delimiter=';')
        self.category2id = category2id
        self.sr = CONFIG["sample_rate"]
        self.target_len = self.sr * 5
        self.has_segments = 'start_time' in self.df.columns and 'end_time' in self.df.columns
        print(f"  Loaded {len(self.df)} test samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['fullfilename']
        category = row['categories']
        label_id = self.category2id.get(category, -1)
        
        chunks = []
        try:
            if self.has_segments and pd.notna(row.get('start_time')):
                y, _ = librosa.load(path, sr=self.sr, mono=True,
                                   offset=row['start_time'], duration=row['segment_duration'])
            else:
                y, _ = librosa.load(path, sr=self.sr, mono=True)
            
            if len(y) > 0:
                if len(y) < self.target_len:
                    padded = np.zeros(self.target_len, dtype=np.float32)
                    padded[:len(y)] = y
                    chunks.append(padded)
                else:
                    stride = int(self.target_len * 0.5)
                    for start in range(0, len(y) - self.target_len + 1, stride):
                        chunks.append(y[start:start + self.target_len])
                    if len(chunks) == 0:
                        chunks.append(y[-self.target_len:])
            else:
                chunks.append(np.zeros(self.target_len, dtype=np.float32))
        except Exception as e:
            print(f"\n❌ Audio error at {idx} for {path}: {e}")
            chunks.append(np.zeros(self.target_len, dtype=np.float32))
        
        return {"audio": chunks, "label": label_id}


def test_collate_fn(batch):
    audio_arrays = []
    all_labels = []
    for item in batch:
        if item["label"] is None or item["label"] < 0:
            continue
        audio_arrays.append(item["audio"])
        all_labels.append(item["label"])
    if len(audio_arrays) == 0:
        return None, None
    return audio_arrays, torch.tensor(all_labels, dtype=torch.long)


# ==========================================
# LABEL MAPPING
# ==========================================

def build_label_mapping(train_csv, test_csv):
    df_train = pd.read_csv(train_csv, delimiter=';')
    df_test = pd.read_csv(test_csv, delimiter=';')
    
    all_categories = set(df_train['categories'].unique()) | set(df_test['categories'].unique())
    all_categories = sorted(list(all_categories))
    
    category2id = {cat: idx for idx, cat in enumerate(all_categories)}
    id2category = {idx: cat for cat, idx in category2id.items()}
    num_classes = len(all_categories)
    
    print(f"✓ Dataset has {num_classes} classes")
    print(f"  Train samples: {len(df_train)}")
    print(f"  Test samples:  {len(df_test)}")
    
    return num_classes, category2id, id2category


# ==========================================
# INFERENCE
# ==========================================

def run_inference(model, test_loader, feature_extractor, num_classes, device):
    """Run inference and compute metrics with timing."""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    print("\n🔊 Running inference on test set...")
    inference_start = time.perf_counter()
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            if batch_data[0] is None:
                continue
            
            audio_arrays, labels = batch_data
            batch_probs = []
            
            for i, audio_chunks in enumerate(audio_arrays):
                if not isinstance(audio_chunks, list):
                    audio_chunks = [audio_chunks]
                
                chunk_probs = []
                for chunk in audio_chunks:
                    inputs = feature_extractor([chunk], padding=True, return_tensors="pt")
                    inputs = inputs.to(device)
                    
                    outputs = model(inputs)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    probs = torch.softmax(logits, dim=1)
                    chunk_probs.append(probs.cpu())
                
                # Max voting across chunks
                final_probs, _ = torch.max(torch.cat(chunk_probs), dim=0)
                batch_probs.append(final_probs)
            
            batch_probs = torch.stack(batch_probs)
            all_probs.append(batch_probs.numpy())
            all_preds.extend(torch.argmax(batch_probs, dim=1).tolist())
            all_labels.extend(labels.tolist())
    
    inference_time = time.perf_counter() - inference_start
    
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Top-3 accuracy
    top3_preds = np.argsort(all_probs, axis=1)[:, -3:][:, ::-1]
    top3_correct = np.sum([all_labels[i] in top3_preds[i] for i in range(len(all_labels))])
    top3_acc = top3_correct / len(all_labels)
    
    # cmAP and AUROC
    y_true_bin = label_binarize(all_labels, classes=range(num_classes))
    
    try:
        cmAP = average_precision_score(y_true_bin, all_probs, average='macro')
    except:
        cmAP = 0.0
    
    try:
        auroc = roc_auc_score(y_true_bin, all_probs, multi_class='ovr', average='macro')
    except:
        auroc = 0.0
    
    return accuracy, top3_acc, cmAP, auroc, inference_time, len(all_labels)


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
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    projection_params = sum(p.numel() for p in model.projection.parameters())
    proto_params = sum(p.numel() for p in model.proto_layer.parameters())
    model_size_mb = total_params * 4 / (1024 ** 2)
    
    print(f"\n  Parameter Breakdown:")
    print(f"    Backbone (frozen):  {backbone_params:,}")
    print(f"    Projection layer:   {projection_params:,}")
    print(f"    Prototype layer:    {proto_params:,}")
    print(f"    ─────────────────────────────")
    print(f"    Total params:       {total_params:,}")
    print(f"    Trainable params:   {trainable_params:,}")
    print(f"    Model size (FP32):  {model_size_mb:.1f} MB")
    
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
        print(f"\n  GPU Memory:")
        print(f"    Peak GPU memory:    {peak_mem:.1f} MB")
    
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
    
    print(f"\n  Inference Speed:")
    print(f"    Avg inference time: {avg_time * 1000:.2f} ms")
    print(f"    Throughput:         {1 / avg_time:.1f} samples/sec")
    
    # --- FLOPs (optional) ---
    try:
        from thop import profile as thop_profile
        flops, _ = thop_profile(model, inputs=(dummy_input,), verbose=False)
        print(f"    FLOPs:              {flops / 1e9:.2f} GFLOPs")
    except ImportError:
        print(f"    FLOPs:              (install 'thop': pip install thop)")
    except Exception as e:
        print(f"    FLOPs:              (could not measure: {e})")
    
    print(f"{'=' * 60}\n")
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "backbone_params": backbone_params,
        "projection_params": projection_params,
        "proto_params": proto_params,
        "model_size_mb": model_size_mb,
        "avg_inference_ms": avg_time * 1000,
        "throughput_per_sec": 1 / avg_time,
    }


# ==========================================
# MAIN
# ==========================================

def main():
    print("\n" + "=" * 60)
    print("PROTOTYPE MODEL TESTING WITH BENCHMARKS")
    print("=" * 60 + "\n")
    
    device = torch.device(CONFIG["device"])
    
    # Build label mapping
    print(" Building label mapping...")
    num_classes, category2id, id2category = build_label_mapping(
        CONFIG["train_csv"], CONFIG["test_csv"]
    )
    
    # Load model
    print(f"\n Loading prototype model...")
    model = AudioProtoPNetClassifier(CONFIG["base_model"], num_classes)
    
    checkpoint = torch.load(CONFIG["model_checkpoint"], map_location=CONFIG["device"])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    print(f" Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f" Val loss: {checkpoint.get('val_loss', '?')}, Val acc: {checkpoint.get('val_acc', '?')}")
    
    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        CONFIG["base_model"], trust_remote_code=True
    )
    
    # Run benchmark
    benchmark_results = benchmark_model(model, feature_extractor, device)
    
    # Create test dataset
    print("\n Loading test dataset...")
    test_dataset = TestDataset(CONFIG["test_csv"], category2id)
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, collate_fn=test_collate_fn, num_workers=0
    )
    
    # Run inference
    accuracy, top3_acc, cmAP, auroc, inference_time, num_samples = run_inference(
        model, test_loader, feature_extractor, num_classes, device
    )
    
    # Print results
    print(f"\n{'=' * 60}")
    print(f"TEST RESULTS:")
    print(f"  Top-1 Accuracy: {accuracy:.4f}")
    print(f"  Top-3 Accuracy: {top3_acc:.4f}")
    print(f"  cmAP:           {cmAP:.4f}")
    print(f"  AUROC:          {auroc:.4f}")
    print(f"{'=' * 60}")
    
    print(f"\nTIMING:")
    print(f"  Total inference time:  {inference_time:.1f}s")
    print(f"  Avg per sample:        {inference_time / max(num_samples, 1) * 1000:.1f}ms")
    print(f"  Samples evaluated:     {num_samples}")
    
    print(f"\nBENCHMARK SUMMARY:")
    print(f"  Total params:          {benchmark_results['total_params']:,}")
    print(f"    Backbone (frozen):   {benchmark_results['backbone_params']:,}")
    print(f"    Projection:          {benchmark_results['projection_params']:,}")
    print(f"    Prototypes:          {benchmark_results['proto_params']:,}")
    print(f"  Trainable params:      {benchmark_results['trainable_params']:,}")
    print(f"  Model size (FP32):     {benchmark_results['model_size_mb']:.1f} MB")
    print(f"  Avg inference time:    {benchmark_results['avg_inference_ms']:.2f} ms")
    print(f"  Throughput:            {benchmark_results['throughput_per_sec']:.1f} samples/sec")
    print(f"{'=' * 60}")
    
    # Save results
    results = {
        "metrics": {
            "top1_accuracy": float(accuracy),
            "top3_accuracy": float(top3_acc),
            "cmAP": float(cmAP),
            "auroc": float(auroc),
        },
        "timing": {
            "total_inference_time_s": float(inference_time),
            "avg_per_sample_ms": float(inference_time / max(num_samples, 1) * 1000),
            "num_samples": num_samples,
        },
        "benchmark": {
            "total_params": benchmark_results["total_params"],
            "trainable_params": benchmark_results["trainable_params"],
            "backbone_params": benchmark_results["backbone_params"],
            "projection_params": benchmark_results["projection_params"],
            "proto_params": benchmark_results["proto_params"],
            "model_size_mb": benchmark_results["model_size_mb"],
            "avg_inference_ms": benchmark_results["avg_inference_ms"],
            "throughput_per_sec": benchmark_results["throughput_per_sec"],
        },
        "config": {
            "model_checkpoint": CONFIG["model_checkpoint"],
            "test_csv": CONFIG["test_csv"],
            "protos_per_class": CONFIG["protos_per_class"],
            "num_classes": num_classes,
        }
    }
    
    output_path = os.path.join(CONFIG["output_dir"], "test_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved to {output_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
