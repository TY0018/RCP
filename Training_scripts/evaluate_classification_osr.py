import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import librosa
import os
import math
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoFeatureExtractor
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Model & Data Paths
    "model_checkpoint": "/home/users/ntu/ytong005/RCP/trained_classifier_proto/best_new_model.pth",
    "base_model": "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
    "known_species_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_train.csv",
    "test_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/open_test1.csv",
    
    # Threshold Configuration
    "threshold_json_path": "RCP/open_set_results_proto/metrics.json", 
    "use_min_distance": True,
    "use_max_softmax": True,
    
    # Output file
    "output_dir": "/home/users/ntu/ytong005/RCP/proto_osr_results/dist_softmax",
    "output_csv": "/home/users/ntu/ytong005/RCP/proto_osr_results/dist_softmax/final_open_set_predictions.csv",
    
    # Model parameters (must match training)
    "protos_per_class": 5,
    "sample_rate": 32000,
    "batch_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ==========================================
# MODEL DEFINITIONS
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
        
        self.projected_dim = 512
        self.proto_layer = MultiPrototypeLayer(
            self.projected_dim, num_classes, CONFIG["protos_per_class"]
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
    
    def forward(self, x, return_features=False):
        feats = self.extract_features(x)
        logits, min_distances = self.proto_layer(feats)
        
        if return_features:
            return logits, min_distances, feats
        return logits, min_distances

# ==========================================
# DATASET
# ==========================================

class InferenceDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, delimiter=';')
        self.sr = CONFIG["sample_rate"]
        self.target_len = self.sr * 5
        self.has_segments = 'start_time' in self.df.columns
        print(f"Loaded {len(self.df)} samples for inference.")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['fullfilename']
        
        # Load audio (with segment bounds if available)
        try:
            if self.has_segments and 'start_time' in row and not pd.isna(row['start_time']):
                y, _ = librosa.load(path, sr=self.sr, mono=True,
                                   offset=row['start_time'], duration=row['segment_duration'])
            else:
                y, _ = librosa.load(path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            y = np.zeros(self.target_len, dtype=np.float32)
        
        # Split into chunks
        chunks = []
        if len(y) > 0:
            if len(y) < self.target_len:
                padded = np.zeros(self.target_len, dtype=np.float32)
                padded[:len(y)] = y
                chunks.append(padded)
            else:
                stride = int(self.target_len * 0.5)
                for start in range(0, len(y) - self.target_len + 1, stride):
                    chunks.append(y[start : start + self.target_len])
                if len(chunks) == 0:
                    chunks.append(y[-self.target_len:])
        else:
            chunks.append(np.zeros(self.target_len, dtype=np.float32))
        
        # Keep original category if we want to save it to output
        true_category = row['categories'] if 'categories' in row else "Unknown"
            
        return {
            "path": path,
            "audio": chunks,
            "true_category": true_category
        }

def collate_fn(batch):
    return {
        "path": [item["path"] for item in batch],
        "audio": [item["audio"] for item in batch],
        "true_category": [item["true_category"] for item in batch]
    }

# ==========================================
# UTILS
# ==========================================

def compute_min_distance_score(min_distances):
    """
    Returns negative distance: higher = closer to a known prototype = more likely known.
    """
    closest_dist, _ = torch.min(min_distances, dim=1)
    return -closest_dist.cpu().numpy()

def load_threshold(json_path):
    """
    Load OSR thresholds from the metrics JSON file.
    Returns (min_distance_threshold, max_softmax_threshold).
    """
    print(f"Loading thresholds from {json_path}...")
    min_dist_thresh = -25.0  # fallback
    max_soft_thresh = 0.1    # fallback
    try:
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'min_distance' in data:
            min_dist_thresh = data['min_distance']['threshold_95tpr']
        if 'max_softmax' in data:
            max_soft_thresh = data['max_softmax']['threshold_95tpr']
    except Exception as e:
        print(f"Failed to load thresholds: {e}")
    return min_dist_thresh, max_soft_thresh


# ==========================================
# MAIN INFERENCE
# ==========================================

def main():
    print("="*60)
    print("OPEN-SET INFERENCE SCRIPT (MIN DISTANCE THRESHOLD)")
    print("="*60)
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 1. Load known species to map predictions back to text
    print("\n📂 Loading known species mapping...")
    df_train = pd.read_csv(CONFIG["known_species_csv"], delimiter=';')
    known_species = sorted(list(set(df_train['categories'].unique())))
    id2category = {idx: cat for idx, cat in enumerate(known_species)}
    category2id = {cat: idx for idx, cat in enumerate(known_species)}
    num_classes = len(known_species)
    print(f"  ✓ {num_classes} known species identified")
    
    # 2. Load Thresholds
    min_dist_threshold, max_soft_threshold = load_threshold(CONFIG["threshold_json_path"])
    
    active_methods = []
    if CONFIG["use_min_distance"]:
        active_methods.append(f"min_distance (threshold={min_dist_threshold:.4f})")
    if CONFIG["use_max_softmax"]:
        active_methods.append(f"max_softmax (threshold={max_soft_threshold:.4f})")
    
    print(f"  ✓ Active OSR methods: {', '.join(active_methods) if active_methods else 'NONE (no rejection)'}")

    # 3. Load Model
    print(f"\n🤖 Loading model...")
    model = AudioProtoPNetClassifier(CONFIG["base_model"], num_classes)
    checkpoint = torch.load(CONFIG["model_checkpoint"], map_location=CONFIG["device"])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(CONFIG["device"])
    model.eval()
    print("  ✓ Model loaded")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        CONFIG["base_model"], trust_remote_code=True
    )
    
    # 4. Load Dataset
    test_dataset = InferenceDataset(CONFIG["test_csv"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                             shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # 5. Run Inference
    results = []
    
    print("\n🚀 Starting inference on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            if batch is None:
                continue
                
            paths = batch["path"]
            audio_arrays = batch["audio"]
            true_categories = batch["true_category"]
            
            for i in range(len(audio_arrays)):
                audio_chunks = audio_arrays[i]
                if not isinstance(audio_chunks, list):
                    audio_chunks = [audio_chunks]
                
                chunk_logits = []
                chunk_min_dists = []
                
                # Process all chunks (sliding windows) for this single file
                for chunk in audio_chunks:
                    inputs = feature_extractor([chunk], padding=True, return_tensors="pt")
                    inputs = inputs.to(CONFIG["device"])
                    
                    logits, min_distances = model(inputs)
                    chunk_logits.append(logits)
                    chunk_min_dists.append(min_distances)
                
                # Aggregate across chunks (average the logits/distances)
                final_logits = torch.cat(chunk_logits, dim=0).mean(dim=0, keepdim=True)
                final_min_dists = torch.cat(chunk_min_dists, dim=0).mean(dim=0, keepdim=True)
                
                # Calculate the min_distance score (higher is more confident)
                min_dist_score = compute_min_distance_score(final_min_dists)[0]
                
                # Get softmax probabilities for metrics
                probs = torch.softmax(final_logits, dim=1).cpu().numpy().squeeze()
                max_softmax_score = float(np.max(probs))
                
                # Predict the class
                pred_idx = torch.argmax(final_logits, dim=1).item()
                predicted_category = id2category[pred_idx]
                
                # Apply OSR — sample must pass ALL enabled checks
                reject_reasons = []
                if CONFIG["use_min_distance"] and min_dist_score < min_dist_threshold:
                    reject_reasons.append("min_distance")
                if CONFIG["use_max_softmax"] and max_softmax_score < max_soft_threshold:
                    reject_reasons.append("max_softmax")
                
                is_unknown = len(reject_reasons) > 0
                
                # Final prediction logic
                final_prediction = "Unknown (Open-Set)" if is_unknown else predicted_category
                
                # True label index (-1 if not a known species)
                true_cat = true_categories[i]
                true_idx = category2id.get(true_cat, -1)
                is_known_gt = true_idx >= 0
                
                results.append({
                    "filename": paths[i],
                    "true_category": true_cat,
                    "predicted_category": predicted_category,
                    "min_distance_score": min_dist_score,
                    "max_softmax_score": max_softmax_score,
                    "is_unknown": is_unknown,
                    "reject_reasons": ",".join(reject_reasons) if reject_reasons else "",
                    "final_prediction": final_prediction,
                    "true_idx": true_idx,
                    "pred_idx": pred_idx,
                    "is_known_gt": is_known_gt,
                    "probs": probs,
                })

    # 6. Save results
    print(f"\n💾 Saving results to {CONFIG['output_csv']}...")
    # Drop internal fields before saving
    save_df = pd.DataFrame([{
        k: v for k, v in r.items() if k not in ('true_idx', 'pred_idx', 'is_known_gt', 'probs')
    } for r in results])
    save_df.to_csv(CONFIG["output_csv"], index=False)
    
    # Print summary statistics
    total_samples = len(results)
    unknown_samples = sum(1 for r in results if r["is_unknown"])
    print(f"  ✓ Processed {total_samples} samples.")
    print(f"  ✓ Flagged {(unknown_samples/total_samples)*100:.2f}% ({unknown_samples}) as Unknown.")
    
    # ========================================
    # 7. Compute Metrics
    # ========================================
    
    # --- Closed-set metrics (known species only) ---
    known_results = [r for r in results if r["is_known_gt"]]
    
    if len(known_results) > 0:
        cs_true = np.array([r["true_idx"] for r in known_results])
        cs_pred = np.array([r["pred_idx"] for r in known_results])
        cs_probs = np.vstack([r["probs"] for r in known_results])
        
        # Top-1 Accuracy
        cs_t1_acc = accuracy_score(cs_true, cs_pred)
        
        # cmAP and AUROC
        y_true_bin = label_binarize(cs_true, classes=range(num_classes))
        active_classes = [i for i in range(num_classes) if np.sum(y_true_bin[:, i]) > 0]
        
        try:
            cs_cmAP = np.mean([
                average_precision_score(y_true_bin[:, i], cs_probs[:, i])
                for i in active_classes
            ])
        except:
            cs_cmAP = 0.0
        
        try:
            cs_auroc = np.mean([
                roc_auc_score(y_true_bin[:, i], cs_probs[:, i])
                for i in active_classes
                if len(np.unique(y_true_bin[:, i])) > 1
            ])
        except:
            cs_auroc = 0.0
        
        print(f"\n{'='*60}")
        print(f"CLOSED-SET RESULTS (known species only, {len(known_results)} samples):")
        print(f"  Top-1 Accuracy: {cs_t1_acc:.4f}")
        print(f"  cmAP:           {cs_cmAP:.4f}")
        print(f"  AUROC:          {cs_auroc:.4f}")
        print(f"{'='*60}")
    else:
        print("\n⚠️  No known-species samples found — cannot compute closed-set metrics.")
    
    # --- Open-set Top-1 Accuracy ---
    # For known samples: correct if predicted correctly AND not rejected
    # For unknown samples: correct if rejected as unknown
    os_correct = 0
    for r in results:
        if r["is_known_gt"]:
            if not r["is_unknown"] and r["pred_idx"] == r["true_idx"]:
                os_correct += 1
        else:
            if r["is_unknown"]:
                os_correct += 1
    
    os_t1_acc = os_correct / total_samples if total_samples > 0 else 0.0
    
    num_unknown_gt = sum(1 for r in results if not r["is_known_gt"])
    num_known_gt = sum(1 for r in results if r["is_known_gt"])
    
    print(f"\nOPEN-SET RESULTS (all {total_samples} samples, threshold={min_dist_threshold:.4f}):")
    print(f"  Open-set Top-1 Accuracy: {os_t1_acc:.4f}")
    print(f"  Known GT: {num_known_gt}, Unknown GT: {num_unknown_gt}")
    print(f"  Accepted: {total_samples - unknown_samples}, Rejected: {unknown_samples}")
    
    # --- OSR Detection Metrics (min_distance as separator) ---
    if num_known_gt > 0 and num_unknown_gt > 0:
        # Binary labels: 1 = known, 0 = unknown
        osr_y_true = np.array([1 if r["is_known_gt"] else 0 for r in results])
        osr_scores = np.array([r["min_distance_score"] for r in results])
        
        # AUROC
        fpr, tpr, thresholds = roc_curve(osr_y_true, osr_scores)
        osr_auroc = auc(fpr, tpr)
        
        # AUPR
        precision, recall, _ = precision_recall_curve(osr_y_true, osr_scores)
        osr_aupr = auc(recall, precision)
        
        # FPR @ 95% TPR
        idx_95 = np.argmax(tpr >= 0.95)
        fpr_at_95tpr = fpr[idx_95]
        
        print(f"\n  OSR Detection (min_distance score):")
        print(f"    AUROC:         {osr_auroc:.4f}")
        print(f"    AUPR:          {osr_aupr:.4f}")
        print(f"    FPR@95%TPR:    {fpr_at_95tpr:.4f}")
    else:
        print(f"\n  ⚠️  Cannot compute OSR detection metrics (need both known and unknown samples).")
    
    # --- Save Plots ---
    plot_dir = CONFIG["output_dir"]
    os.makedirs(plot_dir, exist_ok=True)
    
    if num_known_gt > 0 and num_unknown_gt > 0:
        # 1. ROC Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC Curve (AUROC = {osr_auroc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
        ax.scatter(fpr_at_95tpr, 0.95, color='red', s=80, zorder=5, label=f'FPR@95%TPR = {fpr_at_95tpr:.4f}')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('OSR ROC Curve (Known vs Unknown)', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        roc_path = os.path.join(plot_dir, 'osr_roc_curve.png')
        fig.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  📊 ROC Curve saved to {roc_path}")
        
        # 2. Precision-Recall Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='#4CAF50', lw=2, label=f'PR Curve (AUPR = {osr_aupr:.4f})')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('OSR Precision-Recall Curve (Known vs Unknown)', fontsize=14)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        pr_path = os.path.join(plot_dir, 'osr_pr_curve.png')
        fig.savefig(pr_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  📊 PR Curve saved to {pr_path}")
        
        # 3. Score Distribution
        known_scores = [r["min_distance_score"] for r in results if r["is_known_gt"]]
        unknown_scores = [r["min_distance_score"] for r in results if not r["is_known_gt"]]
        
        all_scores = known_scores + unknown_scores
        # We take the 1st percentile for the lower bound and a bit above the max for the upper
        lower_limit = np.percentile(all_scores, 1) 
        upper_limit = max(all_scores) * 1.05 if max(all_scores) > 0 else 0.5

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(known_scores, bins=50, alpha=0.6, color='#2196F3', range=(lower_limit, upper_limit), label=f'Known ({len(known_scores)})', density=True)
        ax.hist(unknown_scores, bins=50, alpha=0.6, color='#F44336', range=(lower_limit, upper_limit), label=f'Unknown ({len(unknown_scores)})', density=True)
        ax.axvline(x=min_dist_threshold, color='black', linestyle='--', lw=2, label=f'Threshold = {min_dist_threshold:.4f}')
        ax.set_xlabel('Min Distance Score (higher = more confident)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Score Distribution: Known vs Unknown', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        dist_path = os.path.join(plot_dir, 'osr_score_distribution.png')
        fig.savefig(dist_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  📊 Score Distribution saved to {dist_path}")
    
    print(f"{'='*60}")
    
    print("\nDONE!")

if __name__ == "__main__":
    main()
