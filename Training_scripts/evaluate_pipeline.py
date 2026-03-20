"""
Pipeline: Detection Intervals -> Prototype Classifier -> OSR

Reads predicted bird detection intervals from a CSV, classifies each segment
using the prototype-based model (new_classification.py), and applies open-set
recognition (OSR) using min_distance thresholding to reject unknown classes.

Computes AUROC, cmAP, and Top-1 Accuracy against the ground truth labels.
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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Audio file
    "audio_file": "/home/users/ntu/ytong005/scratch/test20min1.wav",
    
    # Detection results CSV
    "detections_csv": "/home/users/ntu/ytong005/RCP/MatchedDetections.csv",
    
    # Model
    "base_model": "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
    "model_checkpoint": "/home/users/ntu/ytong005/RCP/trained_classifier_proto/best_new_model.pth",
    
    # Known species from training set
    "train_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_train.csv",
    
    # Prototype parameters (must match training)
    "protos_per_class": 5,
    
    "use_min_distance": True,
    "use_max_softmax": True,
    "osr_threshold":  -8.132477760314941,
    "max_softmax_threshold": 0.2237648069858551,
    # Parameters
    "sample_rate": 32000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Output
    "output_dir": "RCP/full_pipeline_results/dist_softmax",
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
    
    # Try loading label2name.json if available for reverse mapping
    label2name_paths = [
        "/home/users/ntu/ytong005/dataset_json/label2name.json",
        os.path.join(os.path.dirname(CONFIG["train_csv"]), "label2name.json"),
    ]
    
    common_to_category = {}
    category_to_common = {}
    
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
                            # Normalize hyphens to spaces for consistent matching
                            normalized_common = common.lower().replace('-', ' ')
                            common_to_category[normalized_common] = genus_species
                            category_to_common[genus_species] = common
            
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
            category_to_common[cat] = cat.replace('_', ' ').title()
    
    return common_to_category, category_to_common, train_categories


# ==========================================
# LOAD & PARSE DETECTIONS
# ==========================================

def load_detections(csv_path, common_to_category, train_categories):
    """
    Parse the detection CSV and build a list of segments with
    start/end times, predicted labels, and ground truth labels.
    """
    df = pd.read_csv(csv_path)
    
    # Check if this is the new matched format
    has_matched = 'pred_start' in df.columns and 'gt_label' in df.columns
    
    col_start = 'pred_start' if has_matched else df.columns[0]
    col_end = 'pred_end' if has_matched else df.columns[1]
    col_label = 'gt_label' if has_matched else df.columns[2]
    
    segments = []
    known_species = set(train_categories)
    
    for idx, row in df.iterrows():
        # Skip False Negatives (no predicted interval to classify)
        if has_matched and 'result' in df.columns and str(row.get('result', '')).strip() == 'FN':
            continue
        
        start = parse_timestamp(row[col_start])
        end = parse_timestamp(row[col_end])
        
        # We skip False Negatives for inference (missing start/end predictions)
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
            
            common_lower = raw_label.lower().replace('-', ' ')
            if common_lower in common_to_category:
                gt_category = common_to_category[common_lower]
                gt_label = raw_label
                is_known = gt_category in known_species
            else:
                # Label not in label2name.json → treat as unknown class
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
    Returns (threshold, osr_auroc, plot_data_dict).
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
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
        return None, None, None
    
    y_true = np.array([1] * len(known_scores) + [0] * len(unknown_scores))
    y_scores = np.array(known_scores + unknown_scores)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    osr_auroc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    osr_aupr = auc(recall, precision)
    
    # Find threshold at 95% TPR
    idx_95 = np.argmax(tpr >= 0.95)
    threshold_95tpr = thresholds[idx_95]
    fpr_at_95tpr = fpr[idx_95]
    
    print(f"\n📊 OSR Threshold Auto-Detection:")
    print(f"  AUROC:           {osr_auroc:.4f}")
    print(f"  AUPR:            {osr_aupr:.4f}")
    print(f"  Threshold @95%TPR: {threshold_95tpr:.4f}")
    print(f"  FPR @95%TPR:     {fpr_at_95tpr:.4f}")
    
    plot_data = {
        'fpr': fpr, 'tpr': tpr, 'auroc': osr_auroc,
        'precision': precision, 'recall': recall, 'aupr': osr_aupr,
        'fpr_at_95tpr': fpr_at_95tpr,
        'known_scores': known_scores, 'unknown_scores': unknown_scores,
    }
    
    return threshold_95tpr, osr_auroc, plot_data

def compute_oscr(known_scores, known_correct_flags, unknown_scores):
    """
    Computes CCR (Correct Classification Rate) and FPR (False Positive Rate) 
    for the OSCR curve.
    """
    thresholds = np.unique(np.concatenate([known_scores, unknown_scores]))
    thresholds = np.sort(thresholds)[::-1]  # Sort descending
    
    ccr = []
    fpr = []
    n_known = len(known_scores)
    n_unknown = len(unknown_scores)
    
    if n_known == 0 or n_unknown == 0:
        return np.array([0]), np.array([0])
        
    for t in thresholds:
        correct_and_accepted = np.sum((known_scores >= t) & known_correct_flags)
        ccr.append(correct_and_accepted / n_known)
        
        false_positives = np.sum(unknown_scores >= t)
        fpr.append(false_positives / n_unknown)
        
    return np.array(fpr), np.array(ccr)

# def compute_open_set_metrics(segments, results, category2id, id2category, 
#                              num_classes, osr_threshold):
#     """
#     Open-set metrics: ALL segments are evaluated.
    
#     - Known species with gt_category → true label = class index
#     - FP/blank/unknown → true label = num_classes ("unknown" class)
    
#     Model prediction:
#     - If OSR score >= threshold → predict the model's top class
#     - If OSR score < threshold  → predict "unknown" (index = num_classes)
    
#     A correct prediction for an FP segment = model rejects it as unknown.
#     A correct prediction for a known segment = model classifies it correctly AND accepts it.
#     """
#     UNKNOWN_IDX = num_classes  # Extra class index for "unknown"
    
#     all_true = []
#     all_pred = []
#     all_probs_extended = []  # num_classes + 1 columns
    
#     accepted = 0
#     rejected = 0
    
#     for seg, res in zip(segments, results):
#         if res is None:
#             continue
        
#         # --- Ground truth ---
#         if seg["gt_category"] is not None and seg["is_known"]:
#             true_idx = category2id[seg["gt_category"]]
#         else:
#             true_idx = UNKNOWN_IDX  # FP, blank, or unknown species
        
#         # --- OSR decision (multi-method) ---
#         osr_score = -res["min_distance"]  # Higher = more confident
#         max_soft = res["max_softmax"]
        
#         should_reject = False
#         if CONFIG["use_min_distance"] and osr_score < osr_threshold:
#             should_reject = True
#         if CONFIG["use_max_softmax"]:
#             ms_thresh = CONFIG["max_softmax_threshold"] if CONFIG["max_softmax_threshold"] is not None else 0.5
#             if max_soft < ms_thresh:
#                 should_reject = True
        
#         if not should_reject:
#             # Accept: use model's classification
#             pred_idx = res["pred_idx"]
#             accepted += 1
#         else:
#             # Reject: predict "unknown"
#             pred_idx = UNKNOWN_IDX
#             rejected += 1
        
#         all_true.append(true_idx)
#         all_pred.append(pred_idx)
        
#         # Build extended probability vector (add unknown "probability")
#         # Use 1 - max_softmax as a proxy for unknown probability
#         known_probs = res["probs"]
#         unknown_prob = 1.0 - res["max_softmax"]
#         extended = np.append(known_probs, unknown_prob)
#         all_probs_extended.append(extended)
    
#     all_true = np.array(all_true)
#     all_pred = np.array(all_pred)
#     all_probs_extended = np.vstack(all_probs_extended)
    
#     total = len(all_true)
#     num_known_gt = np.sum(all_true != UNKNOWN_IDX)
#     num_unknown_gt = np.sum(all_true == UNKNOWN_IDX)
    
#     print(f"\n📊 Open-Set Evaluation ({total} segments):")
#     print(f"  Ground truth: {num_known_gt} known, {num_unknown_gt} unknown/FP")
#     print(f"  Model: {accepted} accepted, {rejected} rejected as unknown")
    
#     # --- Open-set Top-1 Accuracy ---
#     # Correct if: known segment classified correctly AND accepted,
#     #          OR unknown segment rejected
#     os_t1_acc = accuracy_score(all_true, all_pred)
    
#     # --- Break down accuracy ---
#     # Known species: correctly classified AND not rejected
#     known_mask = all_true != UNKNOWN_IDX
#     if np.sum(known_mask) > 0:
#         known_correct = np.sum((all_pred == all_true) & known_mask)
#         known_acc = known_correct / np.sum(known_mask)
#     else:
#         known_acc = 0.0
    
#     # Unknown/FP: correctly rejected
#     unknown_mask = all_true == UNKNOWN_IDX
#     if np.sum(unknown_mask) > 0:
#         unknown_correct = np.sum((all_pred == UNKNOWN_IDX) & unknown_mask)
#         rejection_acc = unknown_correct / np.sum(unknown_mask)
#     else:
#         rejection_acc = 0.0
    
#     # --- Open-set cmAP ---
#     # Include the unknown class as an additional class
#     num_classes_ext = num_classes + 1
#     y_true_bin = label_binarize(all_true, classes=range(num_classes_ext))
    
#     active_classes = [i for i in range(num_classes_ext) if np.sum(y_true_bin[:, i]) > 0]
    
#     try:
#         os_cmAP = np.mean([
#             average_precision_score(y_true_bin[:, i], all_probs_extended[:, i])
#             for i in active_classes
#         ])
#     except:
#         os_cmAP = 0.0
    
#     # --- Open-set AUROC ---
#     try:
#         os_auroc = np.mean([
#             roc_auc_score(y_true_bin[:, i], all_probs_extended[:, i])
#             for i in active_classes
#             if len(np.unique(y_true_bin[:, i])) > 1
#         ])
#     except:
#         os_auroc = 0.0
    
#     return {
#         "os_t1_acc": os_t1_acc,
#         "known_acc": known_acc,
#         "rejection_acc": rejection_acc,
#         "os_cmAP": os_cmAP,
#         "os_auroc": os_auroc,
#         "accepted": accepted,
#         "rejected": rejected,
#         "total": total,
#     }

def compute_open_set_metrics(segments, results, category2id, id2category, 
                             num_classes, osr_threshold):
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, auc
    from sklearn.preprocessing import label_binarize
    import numpy as np
    
    UNKNOWN_IDX = num_classes  
    
    all_true = []
    all_pred = []
    all_probs_extended = []
    
    # Tracking for OSCR and Precision/Recall
    k_correct_flags = []
    k_scores_dist = []
    k_scores_msp = []
    u_scores_dist = []
    u_scores_msp = []
    
    tp = 0  # True Positive: Known, correctly ID'd, accepted
    tn = 0  # True Negative: Unknown, rejected
    fp_noise = 0      # False Positive: Unknown, accepted
    fp_misclass = 0   # False Positive: Known, wrong ID, accepted
    fn = 0  # False Negative: Known, rejected
    
    accepted = 0
    rejected = 0
    
    for seg, res in zip(segments, results):
        if res is None:
            continue
        
        # Ground truth
        is_known_gt = seg["gt_category"] is not None and seg["is_known"]
        true_idx = category2id[seg["gt_category"]] if is_known_gt else UNKNOWN_IDX
        
        # Scores
        osr_score = -res["min_distance"]  
        max_soft = res["max_softmax"]
        pred_idx = res["pred_idx"]
        
        # Dual-Gate Rejection Logic
        should_reject = False
        if CONFIG.get("use_min_distance", True) and osr_score < osr_threshold:
            should_reject = True
        if CONFIG.get("use_max_softmax", False):
            ms_thresh = CONFIG.get("max_softmax_threshold", 0.5)
            if max_soft < ms_thresh:
                should_reject = True
        
        # Track for standard metrics
        if not should_reject:
            final_pred = pred_idx
            accepted += 1
        else:
            final_pred = UNKNOWN_IDX
            rejected += 1
            
        # Track for Precision/Recall
        if is_known_gt:
            k_correct_flags.append(pred_idx == true_idx)
            k_scores_dist.append(osr_score)
            k_scores_msp.append(max_soft)
            
            if not should_reject and pred_idx == true_idx:
                tp += 1
            elif not should_reject and pred_idx != true_idx:
                fp_misclass += 1
            else:
                fn += 1
        else:
            u_scores_dist.append(osr_score)
            u_scores_msp.append(max_soft)
            
            if should_reject:
                tn += 1
            else:
                fp_noise += 1
        
        all_true.append(true_idx)
        all_pred.append(final_pred)
        
        known_probs = res["probs"]
        unknown_prob = 1.0 - max_soft
        all_probs_extended.append(np.append(known_probs, unknown_prob))
    
    # Calculate Precision, Recall, F1, Accuracy
    total_fp = fp_noise + fp_misclass
    num_k = tp + fp_misclass + fn
    num_u = tn + fp_noise
    total = num_k + num_u
    
    precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0
    recall = tp / num_k if num_k > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    total_accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # Calculate OSCR
    k_scores_dist = np.array(k_scores_dist)
    k_scores_msp = np.array(k_scores_msp)
    k_correct_flags = np.array(k_correct_flags)
    u_scores_dist = np.array(u_scores_dist)
    u_scores_msp = np.array(u_scores_msp)
    
    fpr_dist, ccr_dist = compute_oscr(k_scores_dist, k_correct_flags, u_scores_dist)
    fpr_msp, ccr_msp = compute_oscr(k_scores_msp, k_correct_flags, u_scores_msp)
    
    auoscr_dist = auc(fpr_dist, ccr_dist) if len(fpr_dist) > 1 else 0.0
    auoscr_msp = auc(fpr_msp, ccr_msp) if len(fpr_msp) > 1 else 0.0

    # Format return dictionary
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_probs_extended = np.vstack(all_probs_extended)
    
    # Legacy Accuracy calculations (kept for your existing print statements)
    os_t1_acc = accuracy_score(all_true, all_pred)
    known_mask = all_true != UNKNOWN_IDX
    known_acc = np.sum((all_pred == all_true) & known_mask) / np.sum(known_mask) if np.sum(known_mask) > 0 else 0.0
    unknown_mask = all_true == UNKNOWN_IDX
    rejection_acc = np.sum((all_pred == UNKNOWN_IDX) & unknown_mask) / np.sum(unknown_mask) if np.sum(unknown_mask) > 0 else 0.0

    return {
        "os_t1_acc": os_t1_acc,
        "known_acc": known_acc,
        "rejection_acc": rejection_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_accuracy": total_accuracy,
        "auoscr_dist": auoscr_dist,
        "auoscr_msp": auoscr_msp,
        "fpr_dist": fpr_dist,
        "ccr_dist": ccr_dist,
        "fpr_msp": fpr_msp,
        "ccr_msp": ccr_msp,
        "tp": tp, "tn": tn, "fp_noise": fp_noise, "fp_misclass": fp_misclass, "fn": fn,
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
    common_to_category, category_to_common, train_categories = build_common_name_mapping()
    
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
    osr_plot_data = None
    if osr_threshold is None:
        osr_threshold, osr_auroc, osr_plot_data = compute_osr_threshold(segments, results)
    else:
        _, osr_auroc, osr_plot_data = compute_osr_threshold(segments, results)
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
    print(f"\n{'=' * 115}")
    print(f"{'Time':<16} {'Ground Truth':<25} {'Prediction (Scientific + Common)':<40} {'Conf':<8} {'MinDist':<10} {'OSR':<10}")
    print(f"{'-' * 115}")
    
    for seg, res in zip(segments, results):
        if res is None:
            continue
        
        time_str = f"{seg['start']:.1f}-{seg['end']:.1f}s"
        gt = seg['gt_label'][:23] if seg['gt_label'] else "—"
        
        pred_cat = res['pred_category']
        common_name = category_to_common.get(pred_cat, "")
        if common_name:
            pred = f"{pred_cat} ({common_name})"[:38]
        else:
            pred = pred_cat[:38]
            
        conf = f"{res['pred_confidence']:.3f}"
        dist = f"{res['min_distance']:.1f}"
        
        # OSR decision (multi-method)
        osr_score = -res["min_distance"]
        max_soft = res["max_softmax"]
        
        should_reject = False
        if CONFIG["use_min_distance"] and osr_threshold is not None and osr_score < osr_threshold:
            should_reject = True
        if CONFIG["use_max_softmax"]:
            ms_thresh = CONFIG["max_softmax_threshold"] if CONFIG["max_softmax_threshold"] is not None else 0.5
            if max_soft < ms_thresh:
                should_reject = True
        
        osr_decision = "REJECT" if should_reject else "Accept"
        
        # Correctness
        if seg['gt_category'] is not None:
            if should_reject:
                match = "✗ (rejected)"
            elif res['pred_category'] == seg['gt_category']:
                match = "✓"
            else:
                match = "✗"
        else:
            if should_reject:
                match = "✓ (rejected FP)"
            else:
                match = "FP (missed)"
        
        print(f"{time_str:<16} {gt:<25} {pred:<40} {conf:<8} {dist:<10} {osr_decision:<10} {match}")
    
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
    
    # if os_metrics is not None:
    #     print(f"\nOPEN-SET RESULTS (all {os_metrics['total']} segments, threshold={osr_threshold:.4f}):")
    #     print(f"  Overall Accuracy:   {os_metrics['os_t1_acc']:.4f}")
    #     print(f"  Known Species Acc:  {os_metrics['known_acc']:.4f}  (correctly classified & accepted)")
    #     print(f"  Rejection Acc:      {os_metrics['rejection_acc']:.4f}  (FP/unknown correctly rejected)")
    #     print(f"  Open-set cmAP:      {os_metrics['os_cmAP']:.4f}")
    #     print(f"  Open-set AUROC:     {os_metrics['os_auroc']:.4f}")
    #     print(f"  Accepted/Rejected:  {os_metrics['accepted']}/{os_metrics['rejected']}")
    
    # if osr_auroc is not None:
    #     print(f"\nOSR SEPARATION (known vs FP/unknown):")
    #     print(f"  Min Distance AUROC: {osr_auroc:.4f}")
    
    print(f"{'=' * 60}")
    if os_metrics is not None:
        print(f"\nOPEN-SET RESULTS (all {os_metrics['total']} segments, threshold={osr_threshold:.4f}):")
        print(f"  Isolated Min-Dist AUOSCR: {os_metrics['auoscr_dist']:.4f}")
        print(f"  Isolated Max-Soft AUOSCR: {os_metrics['auoscr_msp']:.4f}")
        print(f"  -------------------------------------------")
        print(f"  HYBRID SYSTEM OVERALL METRICS:")
        print(f"    Precision:          {os_metrics['precision']:.4f}")
        print(f"    Recall (CCR):       {os_metrics['recall']:.4f}")
        print(f"    F1-Score:           {os_metrics['f1_score']:.4f}")
        print(f"    Total Accuracy:     {os_metrics['total_accuracy']:.4f}")
        print(f"  -------------------------------------------")
        print(f"  System Breakdown:")
        print(f"    True Positives:     {os_metrics['tp']}")
        print(f"    True Negatives:     {os_metrics['tn']}")
        print(f"    False Positives:    {os_metrics['fp_noise']} (Noise) + {os_metrics['fp_misclass']} (Misclassified)")
        print(f"    False Negatives:    {os_metrics['fn']}")
        print(f"    Accepted/Rejected:  {os_metrics['accepted']}/{os_metrics['rejected']}")
        
        # ========================================
        # --- GENERATE PLOTS ---
        # ========================================
        if 'fpr_dist' in os_metrics and len(os_metrics['fpr_dist']) > 1:
            plot_dir = CONFIG["output_dir"]
            os.makedirs(plot_dir, exist_ok=True)
            
            # 1. OSCR Curve Plot (Hybrid System)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.plot(os_metrics['fpr_dist'], os_metrics['ccr_dist'], color='#2196F3', lw=2, 
                    label=f'Min-Distance (AUOSCR: {os_metrics["auoscr_dist"]:.4f})')
            ax.plot(os_metrics['fpr_msp'], os_metrics['ccr_msp'], color='#FF9800', lw=2, linestyle='--', 
                    label=f'Max-Softmax (AUOSCR: {os_metrics["auoscr_msp"]:.4f})')
            
            hybrid_fpr = os_metrics['fp_noise'] / (os_metrics['tn'] + os_metrics['fp_noise']) if (os_metrics['tn'] + os_metrics['fp_noise']) > 0 else 0
            ax.plot(hybrid_fpr, os_metrics['recall'], marker='*', markersize=15, color='red', 
                    linestyle='None', label='Hybrid System (Dual-Gate)', markeredgecolor='black')
    
            ax.set_xscale('log') 
            ax.set_xlabel('False Positive Rate (log scale)', fontsize=12)
            ax.set_ylabel('Correct Classification Rate', fontsize=12)
            ax.set_title('OSCR Curves: Min-Dist vs. MSP with Hybrid Operating Point', fontsize=14)
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, which="both", alpha=0.3)
            
            oscr_path = os.path.join(plot_dir, 'combined_oscr_comparison.png')
            fig.savefig(oscr_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  📊 Combined OSCR Plot saved to {oscr_path}")
            
            # --- Extract Scores for Distributions ---
            known_dist, unknown_dist = [], []
            known_msp, unknown_msp = [], []
            
            for seg, res in zip(segments, results):
                if res is None: continue
                is_known_gt = seg["gt_category"] is not None and seg["is_known"]
                
                if is_known_gt:
                    known_dist.append(-res["min_distance"])
                    known_msp.append(res["max_softmax"])
                else:
                    unknown_dist.append(-res["min_distance"])
                    unknown_msp.append(res["max_softmax"])

            # 2. Min-Distance Score Distribution Plot
            if known_dist and unknown_dist:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Percentile clipping to ignore extreme distance outliers
                all_dist = known_dist + unknown_dist
                lower_limit = np.percentile(all_dist, 1)
                upper_limit = max(all_dist) * 1.05 if max(all_dist) > 0 else 0.5
                
                ax.hist(known_dist, bins=50, alpha=0.6, color='#2196F3', range=(lower_limit, upper_limit), label=f'Known ({len(known_dist)})', density=True)
                ax.hist(unknown_dist, bins=50, alpha=0.6, color='#F44336', range=(lower_limit, upper_limit), label=f'Unknown ({len(unknown_dist)})', density=True)
                
                if osr_threshold is not None:
                    ax.axvline(x=osr_threshold, color='black', linestyle='--', lw=2, label=f'Dist Threshold = {osr_threshold:.4f}')
                
                ax.set_xlabel('Min Distance Score (higher = more confident)', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title('Score Distribution (Min-Distance): Known vs Unknown', fontsize=14)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                dist_path = os.path.join(plot_dir, 'osr_score_dist_mindist.png')
                fig.savefig(dist_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  📊 Min-Distance Dist saved to {dist_path}")

            # 3. Max-Softmax Score Distribution Plot
            if known_msp and unknown_msp:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Softmax is always 0 to 1, so we can fix the range easily
                ax.hist(known_msp, bins=50, alpha=0.6, color='#FF9800', range=(0, 1), label=f'Known ({len(known_msp)})', density=True)
                ax.hist(unknown_msp, bins=50, alpha=0.6, color='#9C27B0', range=(0, 1), label=f'Unknown ({len(unknown_msp)})', density=True)
                
                ms_thresh = CONFIG.get("max_softmax_threshold", 0.5)
                ax.axvline(x=ms_thresh, color='black', linestyle='--', lw=2, label=f'MSP Threshold = {ms_thresh:.4f}')
                
                ax.set_xlabel('Max Softmax Probability (0.0 to 1.0)', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title('Score Distribution (Max Softmax): Known vs Unknown', fontsize=14)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                msp_path = os.path.join(plot_dir, 'osr_score_dist_maxsoftmax.png')
                fig.savefig(msp_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  📊 Max-Softmax Dist saved to {msp_path}")
    # # Save results
    # output = {
    #     "closed_set": {
    #         "top1_accuracy": float(cs_t1),
    #         "top3_accuracy": float(cs_top3),
    #         "cmAP": float(cs_cmAP),
    #         "auroc": float(cs_auroc),
    #     },
    #     "open_set": os_metrics if os_metrics else {},
    #     "osr": {
    #         "min_distance_auroc": float(osr_auroc) if osr_auroc else None,
    #         "threshold": float(osr_threshold) if osr_threshold else None,
    #     },
    #     "segments": []
    # }
    
    # for seg, res in zip(segments, results):
    #     if res is None:
    #         continue
    #     osr_score = -res["min_distance"]
    #     osr_decision = "accept" if (osr_threshold and osr_score >= osr_threshold) else "reject"
        
    #     output["segments"].append({
    #         "start": seg["start"],
    #         "end": seg["end"],
    #         "gt_label": seg["gt_label"],
    #         "gt_category": seg["gt_category"],
    #         "is_known": seg["is_known"],
    #         "pred_category": res["pred_category"],
    #         "pred_confidence": float(res["pred_confidence"]),
    #         "min_distance": float(res["min_distance"]),
    #         "max_softmax": float(res["max_softmax"]),
    #         "entropy": float(res["entropy"]),
    #         "osr_decision": osr_decision,
    #     })
    
    # output_path = os.path.join(CONFIG["output_dir"], "pipeline_results.json")
    # with open(output_path, 'w') as f:
    #     json.dump(output, f, indent=2)
    
    # print(f"\n💾 Results saved to {output_path}")
    
    # # --- Save Plots ---
    # if osr_plot_data is not None:
    #     plot_dir = CONFIG["output_dir"]
        
    #     # 1. ROC Curve
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.plot(osr_plot_data['fpr'], osr_plot_data['tpr'], color='#2196F3', lw=2,
    #             label=f'ROC Curve (AUROC = {osr_plot_data["auroc"]:.4f})')
    #     ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    #     ax.scatter(osr_plot_data['fpr_at_95tpr'], 0.95, color='red', s=80, zorder=5,
    #                label=f'FPR@95%TPR = {osr_plot_data["fpr_at_95tpr"]:.4f}')
    #     ax.set_xlabel('False Positive Rate', fontsize=12)
    #     ax.set_ylabel('True Positive Rate', fontsize=12)
    #     ax.set_title('OSR ROC Curve (Known vs Unknown)', fontsize=14)
    #     ax.legend(loc='lower right', fontsize=10)
    #     ax.grid(True, alpha=0.3)
    #     roc_path = os.path.join(plot_dir, 'osr_roc_curve.png')
    #     fig.savefig(roc_path, dpi=150, bbox_inches='tight')
    #     plt.close(fig)
    #     print(f"  📊 ROC Curve saved to {roc_path}")
        
    #     # 2. Precision-Recall Curve
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.plot(osr_plot_data['recall'], osr_plot_data['precision'], color='#4CAF50', lw=2,
    #             label=f'PR Curve (AUPR = {osr_plot_data["aupr"]:.4f})')
    #     ax.set_xlabel('Recall', fontsize=12)
    #     ax.set_ylabel('Precision', fontsize=12)
    #     ax.set_title('OSR Precision-Recall Curve (Known vs Unknown)', fontsize=14)
    #     ax.legend(loc='lower left', fontsize=10)
    #     ax.grid(True, alpha=0.3)
    #     pr_path = os.path.join(plot_dir, 'osr_pr_curve.png')
    #     fig.savefig(pr_path, dpi=150, bbox_inches='tight')
    #     plt.close(fig)
    #     print(f"  📊 PR Curve saved to {pr_path}")
        
    #     # 3. Score Distribution
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     ax.hist(osr_plot_data['known_scores'], bins=50, alpha=0.6, color='#2196F3',
    #             label=f'Known ({len(osr_plot_data["known_scores"])})', density=True)
    #     ax.hist(osr_plot_data['unknown_scores'], bins=50, alpha=0.6, color='#F44336',
    #             label=f'Unknown ({len(osr_plot_data["unknown_scores"])})', density=True)
    #     if osr_threshold is not None:
    #         ax.axvline(x=osr_threshold, color='black', linestyle='--', lw=2,
    #                    label=f'Threshold = {osr_threshold:.4f}')
    #     ax.set_xlabel('Min Distance Score (higher = more confident)', fontsize=12)
    #     ax.set_ylabel('Density', fontsize=12)
    #     ax.set_title('Score Distribution: Known vs Unknown', fontsize=14)
    #     ax.legend(fontsize=10)
    #     ax.grid(True, alpha=0.3)
    #     dist_path = os.path.join(plot_dir, 'osr_score_distribution.png')
    #     fig.savefig(dist_path, dpi=150, bbox_inches='tight')
    #     plt.close(fig)
    #     print(f"  📊 Score Distribution saved to {dist_path}")
    
    # print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
