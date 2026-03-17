import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import librosa
import json
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Model & Data
    "model_checkpoint": "/home/users/ntu/ytong005/RCP/trained_classifier_proto/best_new_model.pth",
    "base_model": "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
    "known_species_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/balanced_sg_dataset1/balanced_train.csv",  # Training species
    "test_csv": "/home/users/ntu/ytong005/scratch/sg_bird_dataset/open_test1.csv",  # Mix of known + unknown species
    
    # Output
    "output_dir": "../open_set_results_proto",
    
    # Open-set detection methods, "mahalanobis"
    "methods": ["max_softmax", "entropy", "energy", "odin", "mahalanobis"],
    
    # ODIN parameters
    "odin_temperature": 1000,
    "odin_epsilon": 0.0012,
    
    # Parameters
    "sample_rate": 32000,
    "batch_size": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==========================================
# MODEL WRAPPER
# ==========================================
class AudioProtoPNetClassifier(nn.Module):
    """Classifier head on AudioProtoPNet features."""
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        
        # Load the full pretrained model
        print(f"Loading pretrained model: {base_model_name}")
        full_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        
        # Unwrap the backbone to get the feature extractor
        if hasattr(full_model, "backbone"):
            self.backbone = full_model.backbone
        elif hasattr(full_model, "base_model"):
            self.backbone = full_model.base_model
        else:
            self.backbone = full_model

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.feature_dim = 1024
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, inputs, return_features=False, compute_gradients=False):
        was_training = self.training if compute_gradients else None
        if compute_gradients:
            # For ODIN: we need gradients w.r.t. input
            self.eval()
            
            with torch.enable_grad():
                outputs = self.backbone(inputs, output_hidden_states=True)
        else:
            # Normal inference: block all gradients
            with torch.no_grad():
                outputs = self.backbone(inputs, output_hidden_states=True)
        
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'):
            features = outputs.hidden_states[-1]
        else:
            raise ValueError("Cannot find features in model output")
        
        # Global average pooling
        if features.dim() == 3:    # (Batch, Time, Dim)
            features = features.mean(dim=1)
        elif features.dim() == 4:  # (Batch, Dim, Time, Freq)
            features = features.mean(dim=(2, 3))
        
        # Classification - needs gradients for ODIN
        if compute_gradients:
            logits = self.classifier(features)
            # Restore training state
            if was_training:
                self.train()
        else:
            with torch.no_grad():
                logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits

# ==========================================
# DATASET
# ==========================================

class OpenSetDataset(Dataset):
    def __init__(self, csv_file, category2id, known_species):
        """
        Args:
            csv_file: CSV with test recordings
            category2id: Mapping from category name to ID
            known_species: Set of species names in training set
        """
        self.df = pd.read_csv(csv_file, delimiter=';')
        self.category2id = category2id
        self.known_species = known_species
        self.sr = CONFIG["sample_rate"]
        self.target_len = self.sr * 5
        
        # Check if has segmentation info
        self.has_segments = 'start_time' in self.df.columns
        
        print(f"  Loaded {len(self.df)} samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        path = row['fullfilename']
        category = row['categories']
        
        # Determine if this is a known or unknown species
        is_known = category in self.known_species
        
        if is_known:
            label_id = self.category2id[category]
        else:
            label_id = -1  # Unknown class marker
        
        # Load audio segment
        try:
            if self.has_segments and 'start_time' in row and not pd.isna(row['start_time']):
                start_time = row['start_time']
                segment_duration = row['segment_duration']
                y, _ = librosa.load(path, sr=self.sr, mono=True, 
                                   offset=start_time, duration=segment_duration)
            else:
                y, _ = librosa.load(path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            y = np.zeros(self.target_len, dtype=np.float32)
        
        # Process audio (sliding windows for inference)
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
        
        return {
            "audio": chunks,
            "label": label_id,
            "is_known": is_known,
            "category": category
        }

# ==========================================
# OPEN-SET DETECTION METHODS
# ==========================================

def compute_max_softmax_score(logits):
    """Maximum softmax probability."""
    probs = torch.softmax(logits, dim=1)
    max_prob, _ = torch.max(probs, dim=1)
    return max_prob.cpu().numpy()

def compute_entropy_score(logits):
    """Entropy of softmax distribution (lower = more confident)."""
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=1)
    # Return negative entropy (higher = more confident, like other scores)
    return -entropy.cpu().numpy()

def compute_odin_score(model, inputs, temperature, epsilon):
    """ODIN: Out-of-Distribution detector using temperature scaling and input perturbation."""
    model.eval()
    
    # 1. Require gradients for the input tensor
    inputs = inputs.detach()
    inputs.requires_grad = True
    
    # 2. Forward pass with temperature scaling
    logits = model(inputs, compute_gradients=True)
    scaled_logits = logits / temperature
    
    # 3. Calculate gradient with respect to the input
    max_score, _ = torch.max(scaled_logits, dim=1)
    max_score.backward(torch.ones_like(max_score))
    
    # 4. Apply perturbation
    if inputs.grad is not None:
        gradient_sign = torch.sign(inputs.grad)
        # Apply the noise. Note: this creates a new tensor
        perturbed_inputs = inputs - epsilon * gradient_sign
    else:
        perturbed_inputs = inputs
        print("Warning: No gradient found in ODIN step.")
    
    # 5. Re-compute with perturbed input
    with torch.no_grad():
        logits = model(perturbed_inputs)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        max_prob, _ = torch.max(probs, dim=1)
    
    return max_prob.cpu().numpy()

def compute_energy_score(logits, temperature=1.0):
    """
    Energy-based score (Liu et al., 2020).
    Energy = -T * log(sum(exp(logits/T)))
    
    Lower energy = known (in-distribution)
    Higher energy = unknown (out-of-distribution)
    
    We return negative energy so higher = more confident (consistent with other methods)
    """
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return -energy.cpu().numpy()  # Negative so higher = more confident

def compute_odin_score_hybrid(model, inputs, temperature=1000, epsilon=0.01):
    """
    Hybrid ODIN: Computes gradients only through the classifier head.
    Perturbs the 1024-D features instead of the raw audio.
    """
    # Ensure model stays in evaluation mode (keeps BatchNorm and Dropout stable)
    model.eval()
    
    # Get features without gradients (pass through the frozen backbone)
    with torch.no_grad():
        _, features = model(inputs, return_features=True, compute_gradients=False)

    with torch.enable_grad():
        features = features.detach().clone()
        features.requires_grad = True
        
        # Forward through classifier
        logits = model.classifier(features)
        scaled_logits = logits / temperature
        
        # Get max and backward
        max_score, _ = torch.max(scaled_logits, dim=1)
        max_score.backward(torch.ones_like(max_score))
        
        if features.grad is not None:
            # Perturb features
            feature_grad = features.grad
            perturbed_features = features.detach() - epsilon * torch.sign(feature_grad)
        else:
            print("Warning: ODIN hybrid failed to find gradients.")
            perturbed_features = features.detach()

    # ---------------------------------------------------------
    # Re-forward the perturbed features (Gradients OFF)
    # ---------------------------------------------------------
    with torch.no_grad():
        logits_perturbed = model.classifier(perturbed_features)
        probs = torch.softmax(logits_perturbed / temperature, dim=1)
        max_prob, _ = torch.max(probs, dim=1)
        
    return max_prob.cpu().numpy()


def compute_mahalanobis_score(features, class_means, precision_matrix):
    """Mahalanobis distance-based confidence score."""
    # features: [batch, feature_dim]
    # class_means: [num_classes, feature_dim]
    # precision_matrix: [feature_dim, feature_dim]
    
    min_distances = []
    for feat in features:
        distances = []
        for class_mean in class_means:
            diff = feat - class_mean
            dist = torch.sqrt(diff @ precision_matrix @ diff)
            distances.append(dist.item())
        min_distances.append(-min(distances))  # Negative distance (higher = closer to known class)
    
    return np.array(min_distances)

# ==========================================
# FEATURE EXTRACTION FOR MAHALANOBIS
# ==========================================

def extract_class_statistics(model, train_loader, feature_extractor, num_classes, device):
    """Extract class means and covariance for Mahalanobis distance."""
    model.eval()
    
    # Collect features per class
    class_features = [[] for _ in range(num_classes)]
    
    print("\nExtracting features from training set for Mahalanobis...")
    
    with torch.no_grad():
        for batch_data in tqdm(train_loader, desc="Extracting features"):
            # print(f"batch_data: {batch_data}")
            if batch_data is None:
                continue
            
            audio_arrays, labels = batch_data["audio"], batch_data["label"]
            
            for i, audio_chunks in enumerate(audio_arrays):
                if not isinstance(audio_chunks, list):
                    audio_chunks = [audio_chunks]
                
                # Use first chunk
                inputs = feature_extractor([audio_chunks[0]], padding=True, 
                                         return_tensors="pt")
                inputs = inputs.to(device)
                
                _, features = model(inputs, return_features=True)
                
                label = labels[i].item()
                class_features[label].append(features.cpu())
    
    # Compute class means
    class_means = []
    all_features = []
    
    for class_idx in range(num_classes):
        if len(class_features[class_idx]) > 0:
            class_feats = torch.cat(class_features[class_idx], dim=0)
            class_mean = class_feats.mean(dim=0)
            class_means.append(class_mean)
            all_features.append(class_feats)
    
    class_means = torch.stack(class_means)
    all_features = torch.cat(all_features, dim=0)
    
    # Compute covariance and precision matrix
    cov = torch.cov(all_features.T)
    precision = torch.linalg.pinv(cov)
    
    print(f"  ✓ Extracted statistics for {num_classes} classes")
    
    return class_means.to(device), precision.to(device)

# ==========================================
# EVALUATION
# ==========================================

def evaluate_open_set(model, test_loader, feature_extractor, device, 
                     class_means=None, precision_matrix=None):
    """
    Evaluate open-set classification performance.
    
    Returns:
        Dictionary of scores and labels for each method
    """
    model.eval()
    
    results = {
        "labels": [],  # 1 = known, 0 = unknown
        "max_softmax": [],
        "entropy": [],
        "energy":[],
        "odin": [],
        "mahalanobis": [],
        "predictions": [],
        "true_labels": [],
        "categories": []
    }
    
    print("\nRunning open-set evaluation...")
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            if batch_data is None:
                continue
            
            audio_arrays = batch_data["audio"]
            labels = batch_data["label"]
            is_known = batch_data["is_known"]
            categories = batch_data["category"]
            
            for i in range(len(audio_arrays)):
                audio_chunks = audio_arrays[i]
                if not isinstance(audio_chunks, list):
                    audio_chunks = [audio_chunks]
                
                # Aggregate scores across chunks
                chunk_logits = []
                chunk_features = []
                
                for chunk in audio_chunks:
                    inputs = feature_extractor([chunk], padding=True, return_tensors="pt")
                    inputs = inputs.to(device)
                    
                    # Get logits and features
                    logits, features = model(inputs, return_features=True)
                    chunk_logits.append(logits)
                    chunk_features.append(features)
                
                # Aggregate
                final_logits = torch.cat(chunk_logits, dim=0).mean(dim=0, keepdim=True)
                final_features = torch.cat(chunk_features, dim=0).mean(dim=0, keepdim=True)
                
                # Method 1: Max Softmax
                max_softmax = compute_max_softmax_score(final_logits)
                
                # Method 2: Entropy and energy
                entropy = compute_entropy_score(final_logits)
                energy = compute_energy_score(final_logits, temperature=1.0)
                
                # Method 3: ODIN
                inputs_copy = inputs.clone()
                odin = compute_odin_score_hybrid(model, inputs_copy, 
                                        CONFIG["odin_temperature"], 
                                        CONFIG["odin_epsilon"])
                
                # Method 4: Mahalanobis (if statistics available)
                if class_means is not None and precision_matrix is not None:
                    mahalanobis = compute_mahalanobis_score(
                        final_features, class_means, precision_matrix
                    )
                else:
                    mahalanobis = [0.0]
                
                # Get prediction
                pred = torch.argmax(final_logits, dim=1).item()
                
                # Store results
                results["labels"].append(1 if is_known[i] else 0)
                results["max_softmax"].append(max_softmax[0])
                results["entropy"].append(entropy[0])
                results["energy"].append(energy[0])
                results["odin"].append(odin[0])
                results["mahalanobis"].append(mahalanobis[0])
                results["predictions"].append(pred)
                results["true_labels"].append(labels[i].item())
                results["categories"].append(categories[i])
    
    return results

def compute_metrics(results):
    """Compute open-set detection metrics."""
    y_true = np.array(results["labels"])
    
    metrics = {}
    
    for method in CONFIG["methods"]:
        scores = np.array(results[method])
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        
        # Find threshold at 95% TPR
        idx_95tpr = np.argmax(tpr >= 0.95)
        fpr_at_95tpr = fpr[idx_95tpr]
        threshold_95tpr = thresholds[idx_95tpr]
        
        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall, precision)
        
        metrics[method] = {
            "auroc": roc_auc,
            "fpr_at_95tpr": fpr_at_95tpr,
            "aupr": pr_auc,
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            "threshold_95tpr": threshold_95tpr
        }
    
    return metrics

# ==========================================
# VISUALIZATION
# ==========================================

def plot_roc_curves(metrics, output_dir):
    """Plot ROC curves for all methods."""
    plt.figure(figsize=(10, 8))
    
    for method, data in metrics.items():
        if method in CONFIG["methods"]:
            plt.plot(data["fpr"], data["tpr"], 
                    label=f'{method.replace("_", " ").title()} (AUC = {data["auroc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Open-Set Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved ROC curves")

def plot_score_distributions(results, output_dir):
    """Plot score distributions for known vs unknown."""
    
    methods = CONFIG["methods"]
    num_methods = len(methods)

    num_cols = 2
    num_rows = (num_methods + 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6 * num_rows))
    
    # Ensure axes is always a flat array even if it's 1x1 or 1x2
    if num_methods == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        scores = np.array(results[method])
        labels = np.array(results["labels"])
        
        known_scores = scores[labels == 1]
        unknown_scores = scores[labels == 0]
        
        ax.hist(known_scores, bins=50, alpha=0.5, label='Known', color='blue')
        ax.hist(unknown_scores, bins=50, alpha=0.5, label='Unknown', color='red')
        
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{method.replace("_", " ").title()} Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved score distributions")

# ==========================================
# MAIN
# ==========================================

def collate_fn(batch):
    """Collate function for open-set dataset."""
    return {
        "audio": [item["audio"] for item in batch],
        "label": torch.tensor([item["label"] for item in batch]),
        "is_known": [item["is_known"] for item in batch],
        "category": [item["category"] for item in batch]
    }

def main():
    print("\n" + "="*70)
    print("OPEN-SET CLASSIFICATION EVALUATION")
    print("="*70 + "\n")
    
    # Load known species from training set
    print(" Loading known species from training set...")
    df_train = pd.read_csv(CONFIG["known_species_csv"], delimiter=';')
    known_species = set(df_train['categories'].unique())
    print(f"  ✓ {len(known_species)} known species")
    
    # Create category mappings
    all_categories = sorted(list(known_species))
    category2id = {cat: idx for idx, cat in enumerate(all_categories)}
    
    # Load test set
    print(f"\n Loading test set from {CONFIG['test_csv']}...")
    df_test = pd.read_csv(CONFIG["test_csv"], delimiter=';')
    test_species = set(df_test['categories'].unique())
    
    unknown_species = test_species - known_species
    known_in_test = test_species & known_species
    
    print(f"  Test set species:     {len(test_species)}")
    print(f"  Known species:        {len(known_in_test)}")
    print(f"  Unknown species:      {len(unknown_species)}")
    print(f"  Total test samples:   {len(df_test)}")
    
    # Load model
    print(f"\n Loading model...")
    num_classes = len(known_species)
    model = AudioProtoPNetClassifier(CONFIG["base_model"], num_classes)
    
    checkpoint = torch.load(CONFIG["model_checkpoint"], map_location=CONFIG["device"])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(CONFIG["device"])
    model.eval()
    
    print(f" Model loaded with {num_classes} output classes")
    
    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        CONFIG["base_model"], trust_remote_code=True
    )
    
    # Create dataset
    test_dataset = OpenSetDataset(CONFIG["test_csv"], category2id, known_species)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Extract class statistics for Mahalanobis (optional)
    class_means, precision_matrix = None, None
    if "mahalanobis" in CONFIG["methods"]:
        print(f"\n Computing Mahalanobis statistics from training set...")
        train_dataset = OpenSetDataset(CONFIG["known_species_csv"], category2id, known_species)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                                 collate_fn=collate_fn, num_workers=0)
        class_means, precision_matrix = extract_class_statistics(
            model, train_loader, feature_extractor, num_classes, CONFIG["device"]
        )

    # Run evaluation
    results = evaluate_open_set(model, test_loader, feature_extractor, 
                               CONFIG["device"], class_means, precision_matrix)
    
    # Compute metrics
    print(f"\n Computing metrics...")
    metrics = compute_metrics(results)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Method':<20} {'AUROC':<10} {'FPR@95%TPR':<12} {'AUPR':<10}")
    print(f"{'-'*70}")
    
    for method in CONFIG["methods"]:
        m = metrics[method]
        print(f"{method.replace('_', ' ').title():<20} {m['auroc']:<10.4f} "
              f"{m['fpr_at_95tpr']:<12.4f} {m['aupr']:<10.4f}")
    
    # Save results
    print(f"\n Saving results...")
    
    # Save metrics
    metrics_summary = {
        method: {
            "auroc": float(metrics[method]["auroc"]),
            "fpr_at_95tpr": float(metrics[method]["fpr_at_95tpr"]),
            "aupr": float(metrics[method]["aupr"]),
            "threshold_95tpr": float(metrics[method]["threshold_95tpr"])
        }
        for method in CONFIG["methods"]
    }
    
    with open(os.path.join(CONFIG["output_dir"], "metrics.json"), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Save detailed results
    results_df = pd.DataFrame({
        "category": results["categories"],
        "is_known": results["labels"],
        "true_label": results["true_labels"],
        "prediction": results["predictions"],
        "max_softmax": results["max_softmax"],
        "entropy": results["entropy"],
        "energy": results["energy"],
        "odin": results["odin"],
        "mahalanobis": results["mahalanobis"]
    })
    results_df.to_csv(os.path.join(CONFIG["output_dir"], "detailed_results.csv"), index=False)
    
    # Plot visualizations
    print(f"\n Creating visualizations...")
    plot_roc_curves(metrics, CONFIG["output_dir"])
    plot_score_distributions(results, CONFIG["output_dir"])
    
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {CONFIG['output_dir']}")
    print(f"  - metrics.json")
    print(f"  - detailed_results.csv")
    print(f"  - roc_curves.png")
    print(f"  - score_distributions.png")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()