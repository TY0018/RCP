"""
Pipeline: Detection Intervals -> Prototype Classifier -> OSR

Reads predicted bird detection intervals from a CSV, classifies each segment
using the prototype-based model (new_classification.py), and applies open-set
recognition (OSR) using min_distance thresholding to reject unknown classes.

Computes AUROC, cmAP, and Top-1 Accuracy against the ground truth labels.

Usage:
    python classify_detections.py
"""

import os
import math
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import json
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Audio file
    "audio_file": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/test20min1.wav",
    
    # Detection results CSV
    "detections_csv": "/home/users/ntu/ytong005/AudioProto/Values - Sheet1.csv",
    
    # Model
    "base_model": "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
    "model_checkpoint": "/home/users/ntu/ytong005/RCP/trained_classifier_proto/best_new_model.pth",
    
    # Known species from training set
    "train_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_train.csv",
    
    # Prototype parameters (must match training)
    "protos_per_class": 5,
    
    # OSR threshold — segments with min_distance > threshold are marked "unknown"
    # You can tune this from the evaluate_open_set.py results (threshold_95tpr)
    "osr_method": "min_distance",  # or "max_softmax", "entropy"
    "osr_threshold": None,  # None = auto-determine from data; or set a float value
    
    # Parameters
    "sample_rate": 32000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Output
    "output_dir": "./detection_classification_results",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ==========================================
# MODEL (from new_classification.py)
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
        full_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        
        if hasattr(full_model, "backbone"):
            self.backbone = full_model.backbone
        elif hasattr(full_model, "base_model"):
            self.backbone = full_model.base_model
        else:
            self.backbone = full_model

        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )
        
        self.proto_layer = MultiPrototypeLayer(
            512, num_classes, CONFIG["protos_per_class"]
        )

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
# TIMESTAMP PARSING
# ==========================================

def parse_timestamp(ts_str):
    """
    Parse timestamps like '0:05.59', '12:41.99', '1:02.29' to seconds.
    Format: M:SS.cs or MM:SS.cs
    """
    ts_str = str(ts_str).strip()
    if not ts_str or ts_str == 'nan':
        return None
    try:
        parts = ts_str.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except:
        return None
    return None


# ==========================================
# COMMON NAME -> Genus_species MAPPING
# ==========================================

def build_common_name_mapping():
    """
    Map common names from the CSV (e.g., 'Large Billed Crow') to 
    Genus_species format used in the training dataset (e.g., 'Corvus_macrorhynchos').
    
    This must match the categories in your training CSV.
    """
    # Load training categories to see what format they're in
    df_train = pd.read_csv(CONFIG["train_csv"], delimiter=';')
    train_categories = sorted(df_train['categories'].unique().tolist())
    
    print(f"\n📋 Training set has {len(train_categories)} categories")
    
    # Build mapping from common name (lowercase) -> training category
    # Strategy: extract common name part from training categories  
    # Training format is typically Genus_species
    # We need to match "Large Billed Crow" -> the correct Genus_species
    
    # Try loading label2name.json if available for reverse mapping
    label2name_paths = [
        "/home/users/ntu/ytong005/dataset_json/label2name.json",
        os.path.join(os.path.dirname(CONFIG["train_csv"]), "label2name.json"),
    ]
    
    common_to_category = {}
    
    for path in label2name_paths:
        if os.path.exists(path):
            with open(path) as f:
                label2name = json.load(f)
            
            # Build reverse: "Common Name" (lowercase) -> Genus_species
            for key, value in label2name.items():
                if '_' in key and ' ' in value:
                    # key = "ostric2", value = "Struthio camelus_Common Ostrich"
                    # or key = "Struthio camelus_Common Ostrich", value = "ostric2"
                    pass
                
                # Handle "Genus species_Common Name" -> extract common name
                if '_' in key and ' ' in key:
                    # key = "Struthio camelus_Common Ostrich"
                    parts = key.split('_', 1)
                    if len(parts) == 2 and ' ' in parts[0]:
                        scientific = parts[0]  # "Struthio camelus"
                        common = parts[1]      # "Common Ostrich"
                        genus_species = scientific.replace(' ', '_')
                        
                        if genus_species in train_categories:
                            common_to_category[common.lower()] = genus_species
            
            if common_to_category:
                print(f"  ✓ Loaded {len(common_to_category)} common name mappings from {path}")
                break
    
    # If mapping didn't work well, try direct matching with training categories
    # Build a simple fuzzy match from category names
    if len(common_to_category) < 5:
        print("  ⚠️  Few mappings from label2name.json, trying direct category matching...")
        for cat in train_categories:
            # Genus_species -> "genus species"
            common_guess = cat.replace('_', ' ').lower()
            common_to_category[common_guess] = cat
    
    return common_to_category, train_categories


# ==========================================
# LOAD & PARSE DETECTIONS
# ==========================================

def load_detections(csv_path, common_to_category, train_categories):
    """
    Parse the detection CSV and build a list of segments with
    start/end times, predicted labels, and ground truth labels.
    """
    df = pd.read_csv(csv_path)
    
    # Use first 3 columns: predicted_start, predicted_end, label
    col_start = df.columns[0]  # 'predicted_start'
    col_end = df.columns[1]    # 'predicted_end'
    col_label = df.columns[2]  # 'label'
    
    segments = []
    known_species = set(train_categories)
    
    for idx, row in df.iterrows():
        start = parse_timestamp(row[col_start])
        end = parse_timestamp(row[col_end])
        
        if start is None or end is None:
            continue
        
        # Get ground truth label
        raw_label = str(row[col_label]).strip() if pd.notna(row[col_label]) else ""
        
        # Determine if it's a true detection or false positive
        if raw_label == "" or raw_label == "nan":
            # No bird detected — this is a FP from the detector (no label)
            gt_label = "__NO_BIRD__"
            gt_category = None
            is_known = False
        elif raw_label.startswith("FP"):
            # Explicit FP annotation
            gt_label = "__FALSE_POSITIVE__"
            gt_category = None
            is_known = False
        else:
            # Has a species label — check if it maps to a known category
            # Handle multi-species: "Indian Cuckoo and Black Naped Monarch"
            # Take the first species mentioned
            if " and " in raw_label:
                raw_label = raw_label.split(" and ")[0].strip()
            
            common_lower = raw_label.lower()
            if common_lower in common_to_category:
                gt_category = common_to_category[common_lower]
                gt_label = raw_label
                is_known = gt_category in known_species
            else:
                gt_label = raw_label
                gt_category = None
                is_known = False
        
        segments.append({
            "start": start,
            "end": end,
            "duration": end - start,
            "gt_label": gt_label,
            "gt_category": gt_category,
            "is_known": is_known,
        })
    
    print(f"\n📊 Parsed {len(segments)} detection segments")
    
    # Summary
    labeled = sum(1 for s in segments if s["gt_category"] is not None)
    fp = sum(1 for s in segments if s["gt_label"] in ("__NO_BIRD__", "__FALSE_POSITIVE__"))
    unknown = sum(1 for s in segments if not s["is_known"] and s["gt_label"] not in ("__NO_BIRD__", "__FALSE_POSITIVE__"))
    
    print(f"  Known species segments: {labeled}")
    print(f"  False positive segments: {fp}")
    print(f"  Unknown species segments: {unknown}")
    
    return segments


# ==========================================
# CLASSIFY SEGMENTS
# ==========================================

def classify_segments(segments, audio_path, model, feature_extractor, 
                      category2id, id2category, device):
    """
    Load each detection segment from the audio file, run the prototype
    classifier, and compute OSR scores.
    """
    sr = CONFIG["sample_rate"]
    target_len = sr * 5  # 5 seconds
    
    # Load full audio once
    print(f"\n🔊 Loading audio file: {audio_path}")
    y_full, _ = librosa.load(audio_path, sr=sr, mono=True)
    total_duration = len(y_full) / sr
    print(f"  ✓ Loaded {total_duration:.1f}s of audio")
    
    results = []
    
    model.eval()
    with torch.no_grad():
        for seg in tqdm(segments, desc="Classifying segments"):
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            
            # Clamp to audio bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(y_full), end_sample)
            
            y = y_full[start_sample:end_sample]
            
            if len(y) == 0:
                results.append(None)
                continue
            
            # Pad or use sliding window
            if len(y) < target_len:
                padded = np.zeros(target_len, dtype=np.float32)
                padded[:len(y)] = y
                chunks = [padded]
            else:
                chunks = []
                stride = int(target_len * 0.5)
                for s in range(0, len(y) - target_len + 1, stride):
                    chunks.append(y[s:s + target_len])
                if len(chunks) == 0:
                    chunks.append(y[-target_len:])
            
            # Run model on each chunk
            all_logits = []
            all_min_dists = []
            
            for chunk in chunks:
                inputs = feature_extractor([chunk], padding=True, return_tensors="pt")
                inputs = inputs.to(device)
                
                logits, min_distances = model(inputs)
                all_logits.append(logits.cpu())
                all_min_dists.append(min_distances.cpu())
            
            # Average across chunks
            avg_logits = torch.cat(all_logits, dim=0).mean(dim=0, keepdim=True)
            avg_min_dists = torch.cat(all_min_dists, dim=0).mean(dim=0, keepdim=True)
            
            # Get prediction
            probs = torch.softmax(avg_logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_confidence = probs[0, pred_idx].item()
            pred_category = id2category[pred_idx]
            
            # OSR scores
            max_softmax = probs.max().item()
            
            # Min distance to closest prototype (lower = more confident)
            closest_dist = torch.min(avg_min_dists).item()
            
            # Entropy
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum().item()
            
            results.append({
                "pred_category": pred_category,
                "pred_idx": pred_idx,
                "pred_confidence": pred_confidence,
                "max_softmax": max_softmax,
                "min_distance": closest_dist,
                "entropy": entropy,
                "probs": probs.numpy().squeeze(),
            })
    
    return results


# ==========================================
# METRICS: CLOSED-SET (known species only)
# ==========================================

def compute_closed_set_metrics(segments, results, category2id, num_classes):
    """
    Closed-set metrics: only evaluate segments with known ground truth labels.
    FP/unknown segments are excluded entirely.
    """
    valid_segments = []
    valid_results = []
    
    for seg, res in zip(segments, results):
        if res is None:
            continue
        if seg["gt_category"] is None:
            continue
        valid_segments.append(seg)
        valid_results.append(res)
    
    if len(valid_segments) == 0:
        print("⚠️  No valid segments with ground truth labels!")
        return 0, 0, 0, 0
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    for seg, res in zip(valid_segments, valid_results):
        true_idx = category2id[seg["gt_category"]]
        all_labels.append(true_idx)
        all_preds.append(res["pred_idx"])
        all_probs.append(res["probs"])
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)
    
    t1_acc = accuracy_score(all_labels, all_preds)
    
    top3_preds = np.argsort(all_probs, axis=1)[:, -3:][:, ::-1]
    top3_correct = sum(all_labels[i] in top3_preds[i] for i in range(len(all_labels)))
    top3_acc = top3_correct / len(all_labels)
    
    y_true_bin = label_binarize(all_labels, classes=range(num_classes))
    active_classes = [i for i in range(num_classes) if np.sum(y_true_bin[:, i]) > 0]
    
    try:
        cmAP = np.mean([
            average_precision_score(y_true_bin[:, i], all_probs[:, i])
            for i in active_classes
        ])
    except:
        cmAP = 0.0
    
    try:
        auroc = np.mean([
            roc_auc_score(y_true_bin[:, i], all_probs[:, i])
            for i in active_classes
            if len(np.unique(y_true_bin[:, i])) > 1
        ])
    except:
        auroc = 0.0
    
    return t1_acc, top3_acc, cmAP, auroc


# ==========================================
# METRICS: OPEN-SET (known + unknown class)
# ==========================================

def compute_osr_threshold(segments, results):
    """
    Auto-determine OSR threshold from the data at 95% TPR.
    Uses known-species segments as positives, FP/unknown as negatives.
    Returns (threshold, osr_auroc).
    """
    from sklearn.metrics import roc_curve, auc
    
    known_scores = []
    unknown_scores = []
    
    for seg, res in zip(segments, results):
        if res is None:
            continue
        score = -res["min_distance"]  # Higher = more confident
        
        if seg["gt_category"] is not None and seg["is_known"]:
            known_scores.append(score)
        else:
            unknown_scores.append(score)
    
    if len(known_scores) == 0 or len(unknown_scores) == 0:
        print("⚠️  Cannot compute threshold — need both known and unknown segments")
        return None, None
    
    y_true = np.array([1] * len(known_scores) + [0] * len(unknown_scores))
    y_scores = np.array(known_scores + unknown_scores)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    osr_auroc = auc(fpr, tpr)
    
    # Find threshold at 95% TPR
    idx_95 = np.argmax(tpr >= 0.95)
    threshold_95tpr = thresholds[idx_95]
    fpr_at_95tpr = fpr[idx_95]
    
    print(f"\n📊 OSR Threshold Auto-Detection:")
    print(f"  AUROC:           {osr_auroc:.4f}")
    print(f"  Threshold @95%TPR: {threshold_95tpr:.4f}")
    print(f"  FPR @95%TPR:     {fpr_at_95tpr:.4f}")
    
    return threshold_95tpr, osr_auroc


def compute_open_set_metrics(segments, results, category2id, id2category, 
                             num_classes, osr_threshold):
    """
    Open-set metrics: ALL segments are evaluated.
    
    - Known species with gt_category → true label = class index
    - FP/blank/unknown → true label = num_classes ("unknown" class)
    
    Model prediction:
    - If OSR score >= threshold → predict the model's top class
    - If OSR score < threshold  → predict "unknown" (index = num_classes)
    
    A correct prediction for an FP segment = model rejects it as unknown.
    A correct prediction for a known segment = model classifies it correctly AND accepts it.
    """
    UNKNOWN_IDX = num_classes  # Extra class index for "unknown"
    
    all_true = []
    all_pred = []
    all_probs_extended = []  # num_classes + 1 columns
    
    accepted = 0
    rejected = 0
    
    for seg, res in zip(segments, results):
        if res is None:
            continue
        
        # --- Ground truth ---
        if seg["gt_category"] is not None and seg["is_known"]:
            true_idx = category2id[seg["gt_category"]]
        else:
            true_idx = UNKNOWN_IDX  # FP, blank, or unknown species
        
        # --- OSR decision ---
        osr_score = -res["min_distance"]  # Higher = more confident
        
        if osr_score >= osr_threshold:
            # Accept: use model's classification
            pred_idx = res["pred_idx"]
            accepted += 1
        else:
            # Reject: predict "unknown"
            pred_idx = UNKNOWN_IDX
            rejected += 1
        
        all_true.append(true_idx)
        all_pred.append(pred_idx)
        
        # Build extended probability vector (add unknown "probability")
        # Use 1 - max_softmax as a proxy for unknown probability
        known_probs = res["probs"]
        unknown_prob = 1.0 - res["max_softmax"]
        extended = np.append(known_probs, unknown_prob)
        all_probs_extended.append(extended)
    
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_probs_extended = np.vstack(all_probs_extended)
    
    total = len(all_true)
    num_known_gt = np.sum(all_true != UNKNOWN_IDX)
    num_unknown_gt = np.sum(all_true == UNKNOWN_IDX)
    
    print(f"\n📊 Open-Set Evaluation ({total} segments):")
    print(f"  Ground truth: {num_known_gt} known, {num_unknown_gt} unknown/FP")
    print(f"  Model: {accepted} accepted, {rejected} rejected as unknown")
    
    # --- Open-set Top-1 Accuracy ---
    # Correct if: known segment classified correctly AND accepted,
    #          OR unknown segment rejected
    os_t1_acc = accuracy_score(all_true, all_pred)
    
    # --- Break down accuracy ---
    # Known species: correctly classified AND not rejected
    known_mask = all_true != UNKNOWN_IDX
    if np.sum(known_mask) > 0:
        known_correct = np.sum((all_pred == all_true) & known_mask)
        known_acc = known_correct / np.sum(known_mask)
    else:
        known_acc = 0.0
    
    # Unknown/FP: correctly rejected
    unknown_mask = all_true == UNKNOWN_IDX
    if np.sum(unknown_mask) > 0:
        unknown_correct = np.sum((all_pred == UNKNOWN_IDX) & unknown_mask)
        rejection_acc = unknown_correct / np.sum(unknown_mask)
    else:
        rejection_acc = 0.0
    
    # --- Open-set cmAP ---
    # Include the unknown class as an additional class
    num_classes_ext = num_classes + 1
    y_true_bin = label_binarize(all_true, classes=range(num_classes_ext))
    
    active_classes = [i for i in range(num_classes_ext) if np.sum(y_true_bin[:, i]) > 0]
    
    try:
        os_cmAP = np.mean([
            average_precision_score(y_true_bin[:, i], all_probs_extended[:, i])
            for i in active_classes
        ])
    except:
        os_cmAP = 0.0
    
    # --- Open-set AUROC ---
    try:
        os_auroc = np.mean([
            roc_auc_score(y_true_bin[:, i], all_probs_extended[:, i])
            for i in active_classes
            if len(np.unique(y_true_bin[:, i])) > 1
        ])
    except:
        os_auroc = 0.0
    
    return {
        "os_t1_acc": os_t1_acc,
        "known_acc": known_acc,
        "rejection_acc": rejection_acc,
        "os_cmAP": os_cmAP,
        "os_auroc": os_auroc,
        "accepted": accepted,
        "rejected": rejected,
        "total": total,
    }


# ==========================================
# MAIN
# ==========================================

def main():
    print("\n" + "=" * 60)
    print("DETECTION -> CLASSIFICATION -> OSR PIPELINE")
    print("=" * 60 + "\n")
    
    device = torch.device(CONFIG["device"])
    
    # Build common name mapping
    print("📋 Building species name mapping...")
    common_to_category, train_categories = build_common_name_mapping()
    
    # Show mapping for verification
    print(f"\n  Common Name Mapping Preview:")
    for common, cat in sorted(common_to_category.items())[:10]:
        print(f"    '{common}' -> {cat}")
    if len(common_to_category) > 10:
        print(f"    ... ({len(common_to_category)} total)")
    
    # Build category2id
    all_categories = sorted(train_categories)
    category2id = {cat: idx for idx, cat in enumerate(all_categories)}
    id2category = {idx: cat for cat, idx in category2id.items()}
    num_classes = len(all_categories)
    
    # Load detections
    print(f"\n📂 Loading detections from CSV...")
    segments = load_detections(CONFIG["detections_csv"], common_to_category, train_categories)
    
    # Load model
    print(f"\n🤖 Loading prototype model...")
    model = AudioProtoPNetClassifier(CONFIG["base_model"], num_classes)
    checkpoint = torch.load(CONFIG["model_checkpoint"], map_location=CONFIG["device"])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print(f"  ✓ Model loaded ({num_classes} classes, {CONFIG['protos_per_class']} prototypes/class)")
    
    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        CONFIG["base_model"], trust_remote_code=True
    )
    
    # Classify all segments
    results = classify_segments(
        segments, CONFIG["audio_file"], model, feature_extractor,
        category2id, id2category, device
    )
    
    # ========================================
    # A. Closed-set metrics (known species only)
    # ========================================
    cs_t1, cs_top3, cs_cmAP, cs_auroc = compute_closed_set_metrics(
        segments, results, category2id, num_classes
    )
    
    # ========================================
    # B. OSR threshold detection
    # ========================================
    osr_threshold = CONFIG["osr_threshold"]
    if osr_threshold is None:
        osr_threshold, osr_auroc = compute_osr_threshold(segments, results)
    else:
        _, osr_auroc = compute_osr_threshold(segments, results)
        print(f"  Using manual threshold: {osr_threshold}")
    
    # ========================================
    # C. Open-set metrics (known + unknown)
    # ========================================
    if osr_threshold is not None:
        os_metrics = compute_open_set_metrics(
            segments, results, category2id, id2category,
            num_classes, osr_threshold
        )
    else:
        os_metrics = None
    
    # ========================================
    # Print detailed per-segment results
    # ========================================
    print(f"\n{'=' * 100}")
    print(f"{'Time':<16} {'Ground Truth':<25} {'Prediction':<25} {'Conf':<8} {'MinDist':<10} {'OSR':<10}")
    print(f"{'-' * 100}")
    
    for seg, res in zip(segments, results):
        if res is None:
            continue
        
        time_str = f"{seg['start']:.1f}-{seg['end']:.1f}s"
        gt = seg['gt_label'][:23] if seg['gt_label'] else "—"
        pred = res['pred_category'][:23]
        conf = f"{res['pred_confidence']:.3f}"
        dist = f"{res['min_distance']:.1f}"
        
        # OSR decision
        osr_score = -res["min_distance"]
        if osr_threshold is not None:
            osr_decision = "Accept" if osr_score >= osr_threshold else "REJECT"
        else:
            osr_decision = "—"
        
        # Correctness
        if seg['gt_category'] is not None:
            if osr_threshold is not None and osr_score < osr_threshold:
                match = "✗ (rejected)"
            elif res['pred_category'] == seg['gt_category']:
                match = "✓"
            else:
                match = "✗"
        else:
            if osr_threshold is not None and osr_score < osr_threshold:
                match = "✓ (rejected FP)"
            else:
                match = "FP (missed)"
        
        print(f"{time_str:<16} {gt:<25} {pred:<25} {conf:<8} {dist:<10} {osr_decision:<10} {match}")
    
    # ========================================
    # Print summary
    # ========================================
    print(f"\n{'=' * 60}")
    print(f"CLOSED-SET RESULTS (known species only, {sum(1 for s in segments if s['gt_category'] is not None)} segments):")
    print(f"  Top-1 Accuracy: {cs_t1:.4f}")
    print(f"  Top-3 Accuracy: {cs_top3:.4f}")
    print(f"  cmAP:           {cs_cmAP:.4f}")
    print(f"  AUROC:          {cs_auroc:.4f}")
    print(f"{'=' * 60}")
    
    if os_metrics is not None:
        print(f"\nOPEN-SET RESULTS (all {os_metrics['total']} segments, threshold={osr_threshold:.4f}):")
        print(f"  Overall Accuracy:   {os_metrics['os_t1_acc']:.4f}")
        print(f"  Known Species Acc:  {os_metrics['known_acc']:.4f}  (correctly classified & accepted)")
        print(f"  Rejection Acc:      {os_metrics['rejection_acc']:.4f}  (FP/unknown correctly rejected)")
        print(f"  Open-set cmAP:      {os_metrics['os_cmAP']:.4f}")
        print(f"  Open-set AUROC:     {os_metrics['os_auroc']:.4f}")
        print(f"  Accepted/Rejected:  {os_metrics['accepted']}/{os_metrics['rejected']}")
    
    if osr_auroc is not None:
        print(f"\nOSR SEPARATION (known vs FP/unknown):")
        print(f"  Min Distance AUROC: {osr_auroc:.4f}")
    
    print(f"{'=' * 60}")
    
    # Save results
    output = {
        "closed_set": {
            "top1_accuracy": float(cs_t1),
            "top3_accuracy": float(cs_top3),
            "cmAP": float(cs_cmAP),
            "auroc": float(cs_auroc),
        },
        "open_set": os_metrics if os_metrics else {},
        "osr": {
            "min_distance_auroc": float(osr_auroc) if osr_auroc else None,
            "threshold": float(osr_threshold) if osr_threshold else None,
        },
        "segments": []
    }
    
    for seg, res in zip(segments, results):
        if res is None:
            continue
        osr_score = -res["min_distance"]
        osr_decision = "accept" if (osr_threshold and osr_score >= osr_threshold) else "reject"
        
        output["segments"].append({
            "start": seg["start"],
            "end": seg["end"],
            "gt_label": seg["gt_label"],
            "gt_category": seg["gt_category"],
            "is_known": seg["is_known"],
            "pred_category": res["pred_category"],
            "pred_confidence": float(res["pred_confidence"]),
            "min_distance": float(res["min_distance"]),
            "max_softmax": float(res["max_softmax"]),
            "entropy": float(res["entropy"]),
            "osr_decision": osr_decision,
        })
    
    output_path = os.path.join(CONFIG["output_dir"], "pipeline_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n Results saved to {output_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
