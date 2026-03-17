import torch
import pandas as pd
import numpy as np
import librosa
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoFeatureExtractor
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "csv_path": "/home/users/ntu/ytong005/VAD/javad/JavadPreds_top20_for_classification.csv",
    "model_path": "model_hard_mined_new.pth", 
    "base_model": "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
    "dataset_json": "/home/users/ntu/ytong005/dataset_json/xcm_ebird.json",
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sample_rate": 32000
}

# ==========================================
# 2. MAPPING: DATASET -> MODEL INDICES
# ==========================================
def build_label_mapping(dataset_json_path, csv_path, model_label2id):
    """
    Finds which Model Indices correspond to the Dataset's classes.
    """
    print("Building Target Mapping...")
    
    # Load Dataset Map (Dataset ID -> Species Code)
    with open(dataset_json_path, 'r') as f:
        ds_data = json.load(f)
        ds_id2label = {int(k): v for k, v in ds_data['id2label'].items()}

    # Map: Dataset_ID (e.g. 40) -> Model_Index (e.g. 55)
    dataset_id_to_model_id = {}
    target_model_indices = [] # List of the 20 indices we care about

    df = pd.read_csv(csv_path)
    # Convert to numeric, forcing errors to NaN, then drop them
    df['clean_cat'] = pd.to_numeric(df['categories'], errors='coerce')
    valid_df = df.dropna(subset=['clean_cat'])
    valid_df['clean_cat'] = valid_df['clean_cat'].astype(int)
    present_ids = sorted(valid_df['clean_cat'].unique())

    for ds_id, species_code in ds_id2label.items():
        if species_code in model_label2id:
            model_idx = model_label2id[species_code]
            dataset_id_to_model_id[ds_id] = model_idx
            if ds_id in present_ids:
                target_model_indices.append(model_idx)
        else:
            print(f"Warning: Dataset species '{species_code}' not found in Model training classes.")
    print(f"Number of present_ids in csv: {len(present_ids)}")
    print(f"Found {len(target_model_indices)} matching classes between Dataset and Model.")
    return dataset_id_to_model_id, sorted(list(set(target_model_indices)))

# ==========================================
# 3. DATASET CLASS (UPDATED FOR VOTING)
# ==========================================
class InferenceDataset(Dataset):
    def __init__(self, csv_file, feature_extractor, id_map):
        self.df = pd.read_csv(csv_file)
        self.feature_extractor = feature_extractor
        self.sr = CONFIG["sample_rate"]
        self.id_map = id_map
        self.target_len = self.sr * 5  # 5 seconds
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['fullfilename']
        raw_category = row['categories'] 
        overlap = row['overlap_percentage']
        
        label_id = -1 
        
        # Resolve Label
        if str(raw_category) != "NA":
            try:
                ds_id = int(float(raw_category)) 
                if ds_id in self.id_map:
                    label_id = self.id_map[ds_id]
            except ValueError:
                pass 
        
        chunks = []
        
        try:
            y, sr = librosa.load(path, sr=self.sr, mono=True)
            
            if len(y) > 0:
                # CASE 1: Audio is SHORTER than 5s -> Pad Once (Safe method)
                if len(y) < self.target_len:
                    padded = np.zeros(self.target_len, dtype=np.float32)
                    padded[:len(y)] = y
                    chunks.append(padded)
                    
                # CASE 2: Audio is LONGER than 5s -> Sliding Window
                else:
                    # Stride: Move forward by 2.5s (50% overlap)
                    stride = int(self.target_len * 0.5) 
                    
                    # Generate windows
                    for start in range(0, len(y) - self.target_len + 1, stride):
                        chunk = y[start : start + self.target_len]
                        chunks.append(chunk)
                    
                    # Safety: If we missed the tail or loop didn't run, add the very end
                    if len(chunks) == 0 or (len(y) - self.target_len) % stride != 0:
                        chunks.append(y[-self.target_len:]) # Capture the end
            else:
                # Empty file fallback
                chunks.append(np.zeros(self.target_len, dtype=np.float32))

        except Exception as e:
            # print(f"Error loading {path}: {e}")
            chunks.append(np.zeros(self.target_len, dtype=np.float32))
            
        # Return list of chunks. We convert to list of numpy arrays.
        return {
            "audio": chunks, 
            "label_id": label_id, 
            "overlap": overlap
        }

# NOTE: We NO LONGER NEED the 'collate_fn' because we will handle 
# batching manually in the loop (batch_size=1).

def collate_fn(batch, feature_extractor):
    audio_list = [x["audio"] for x in batch]
    inputs = feature_extractor(audio_list, padding=True, return_tensors="pt")
    label_ids = torch.tensor([x["label_id"] for x in batch]).long()
    overlaps = torch.tensor([x["overlap"] for x in batch]).float()
    return inputs, label_ids, overlaps

# ==========================================
# 4. EVALUATION
# ==========================================
def run_evaluation():
    print("Loading Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model"], trust_remote_code=True
    )
    
    # Load weights
    state_dict = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Load Status: {msg}")
    
    model.to(CONFIG["device"])
    model.eval()
    
    ds_to_model_map, target_indices = build_label_mapping(CONFIG["dataset_json"], CONFIG["csv_path"], model.config.label2id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(CONFIG["base_model"], trust_remote_code=True)
    
    ds = InferenceDataset(CONFIG["csv_path"], feature_extractor, ds_to_model_map)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    
    all_preds = []
    
    # --- CONFIGURATION FOR MEMORY SAFETY ---
    CHUNK_BATCH_SIZE = 16  # Process max 16 windows at a time. Reduce to 8 if OOM persists.
    
    print(f"Running Inference on {len(ds)} files with Sliding Window Voting...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            # 1. Unpack chunks
            raw_chunks = [chunk.squeeze().numpy() for chunk in batch['audio']]
            label_id = batch['label_id'].item()
            overlap = batch['overlap'].item()
            
            # 2. Process Chunks in Mini-Batches (Looping to save memory)
            file_chunk_probs = []
            
            # Loop through the list of chunks in steps of CHUNK_BATCH_SIZE
            for i in range(0, len(raw_chunks), CHUNK_BATCH_SIZE):
                # Slice a small batch (e.g., chunks 0-15)
                mini_batch = raw_chunks[i : i + CHUNK_BATCH_SIZE]
                
                # Extract features for this small batch
                inputs = feature_extractor(mini_batch, padding=True, return_tensors="pt")
                inputs = inputs.to(CONFIG["device"])
                
                # Run Model
                outputs = model(inputs)
                
                # Get probabilities and move to CPU immediately to free GPU
                probs = torch.softmax(outputs.logits, dim=1).cpu()
                file_chunk_probs.append(probs)
                
                # Optional: Clear cache if memory is extremely tight
                # torch.cuda.empty_cache() 

            # 3. Aggregate Results
            if len(file_chunk_probs) > 0:
                all_probs_tensor = torch.cat(file_chunk_probs, dim=0)
                
                # 4. VOTING (Max Pooling)
                # Take max confidence across all time windows
                vote_probs, _ = torch.max(all_probs_tensor, dim=0)
                final_probs_np = vote_probs.numpy()
            else:
                # Fallback for empty/failed audio
                final_probs_np = np.zeros(len(model.config.id2label))

            # 5. Get Prediction
            pred_id = np.argmax(final_probs_np)
            pred_str = model.config.id2label.get(pred_id, str(pred_id))

            all_preds.append({
                "true_id": label_id,
                "pred_id": pred_id,
                "pred_str": str(pred_str), 
                "confidence": np.max(final_probs_np),
                "overlap": overlap,
                "probs": final_probs_np 
            })
                
    return pd.DataFrame(all_preds), target_indices, model.config.id2label

# ==========================================
# 5. SUBSET METRIC CALCULATION
# ==========================================
def analyze_results(df, target_indices):
    """
    df: DataFrame containing 'true_id', 'pred_id', 'probs', and 'overlap'
    target_indices: List of valid class IDs (e.g., the specific birds in your subset)
    """
    
    # 1. FILTERING: Keep only samples belonging to our target subset
    df_valid = df[df['true_id'].isin(target_indices)].copy()
    
    if len(df_valid) == 0:
        print(" No valid bird samples found for the target indices.")
        return

    # Prepare Data
    y_true = df_valid['true_id'].values
    # Stack probabilities: Shape (N_samples, Total_Model_Classes)
    full_probs = np.stack(df_valid['probs'].values) 

    # 1. Slice: Get the cols for your ~16-20 birds. Shape (N, Subset_Size)
    subset_probs = full_probs[:, target_indices]
    
    # 2. Sum: Calculate the total probability mass assigned to this subset
    # (e.g., usually tiny numbers like 0.002)
    row_sums = subset_probs.sum(axis=1, keepdims=True)
    
    # 3. Divide: Renormalize so rows sum to 1.0
    # +1e-12 prevents division by zero if a row is pure 0
    renorm_probs = subset_probs / (row_sums + 1e-12)
    
    # Calculate new Confidence (Max of the renormalized probs)
    new_confidences = np.max(renorm_probs, axis=1)
    
    # ---------------------------------------------------------
    # 2. CLOSED-SET ACCURACY (The Fix)
    # ---------------------------------------------------------
    subset_probs = full_probs[:, target_indices]
    
    relative_preds = np.argmax(renorm_probs, axis=1)
    
    # Map back to original Class IDs
    target_indices_arr = np.array(target_indices)
    y_pred_closed = target_indices_arr[relative_preds]
    
    acc = accuracy_score(y_true, y_pred_closed)
    
    print("\n" + "="*60)
    print(f"Closed-Set Accuracy (Restricted to {len(target_indices)} classes): {acc:.4f}")
    print("="*60)

    # ---------------------------------------------------------
    # 3. cmAP and AUROC (One-vs-Rest within Subset)
    # ---------------------------------------------------------
    aps = []
    aucs = []
    valid_targets = 0

    print(f"\n{'Class ID':<10} {'Samples':<10} {'AP':<10} {'AUC':<10}")
    print("-" * 50)
    
    for class_id in target_indices:
        y_true_binary = (y_true == class_id).astype(int)
        
        y_score_class = full_probs[:, class_id]

        if len(np.unique(y_true_binary)) == 2:
            ap = average_precision_score(y_true_binary, y_score_class)
            auc = roc_auc_score(y_true_binary, y_score_class)
            
            aps.append(ap)
            aucs.append(auc)
            valid_targets += 1
            
            print(f"{class_id:<10} {sum(y_true_binary):<10} {ap:.4f}     {auc:.4f}")
        else:
            pass

    print("-" * 50)
    
    if valid_targets > 0:
        macro_map = np.mean(aps)
        macro_auc = np.mean(aucs)
        print(f"Subset cmAP  (Macro-Average): {macro_map:.4f}")
        print(f"Subset AUROC (Macro-Average): {macro_auc:.4f}")
    else:
        print("Could not calculate subset metrics (Not enough class variation).")

    # ---------------------------------------------------------
    # 4. PLOTTING (Accuracy vs Overlap)
    # ---------------------------------------------------------
    # Use the Closed-Set Prediction for the "is_correct" check
    df_valid['is_correct'] = (df_valid['true_id'] == y_pred_closed).astype(int)
    
    # Create bins for overlap (Confidence/Quality metric)
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    df_valid['overlap_bin'] = pd.cut(df_valid['overlap'], bins=bins, labels=labels, include_lowest=True)
    
    # Calculate accuracy per bin
    bin_acc = df_valid.groupby('overlap_bin', observed=False)['is_correct'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=bin_acc, x='overlap_bin', y='is_correct', palette="viridis")
    plt.ylim(0, 1.0)
    plt.title("Closed-Set Accuracy vs. Signal Overlap (Quality)")
    plt.ylabel("Accuracy")
    plt.xlabel("Overlap Score")
    
    # Save and Show
    save_path = "graph_overlap_vs_acc4.png"
    plt.savefig(save_path)
    print(f"\nSaved graph to {save_path}")
    plt.show()

    return acc, macro_map, macro_auc

def analyze_errors_topk(df, target_indices, class_names=None, num_print=20):
    """
    Prints Top-3 probabilities ONLY for misclassified samples.
    
    df: DataFrame with 'true_id', 'probs'
    target_indices: List of valid class IDs
    class_names: (Optional) Dict or List mapping ID -> "Bird Name"
    num_print: Max number of error rows to display
    """
    
    # 1. Filter Valid Data
    df_valid = df[df['true_id'].isin(target_indices)].copy()
    
    if len(df_valid) == 0:
        print("No valid bird samples found.")
        return

    y_true = df_valid['true_id'].values
    full_probs = np.stack(df_valid['probs'].values)
    
    # 2. CLOSED-SET SLICING (Focus only on target birds)
    target_arr = np.array(target_indices)
    subset_probs = full_probs[:, target_indices]
    
    # 3. GET TOP 3 PREDICTIONS
    relative_top3_idx = np.argsort(subset_probs, axis=1)[:, -3:][:, ::-1]
    
    # Map back to REAL Class IDs
    top3_ids = target_arr[relative_top3_idx]
    
    # Get the corresponding probabilities
    top3_probs = np.take_along_axis(subset_probs, relative_top3_idx, axis=1)

    # 4. IDENTIFY ERRORS
    # Top-1 prediction is the first column
    pred_top1 = top3_ids[:, 0]
    
    # Create mask where Prediction != True Label
    error_mask = (pred_top1 != y_true)
    
    # Extract only the error rows
    y_true_errors = y_true[error_mask]
    top3_ids_errors = top3_ids[error_mask]
    top3_probs_errors = top3_probs[error_mask]
    
    total_errors = len(y_true_errors)
    
    print("\n" + "="*90)
    print(f" FOUND {total_errors} ERRORS (out of {len(df_valid)} samples)")
    print("="*90)

    # 5. PRINT DETAILED ERROR TABLE
    if total_errors > 0:
        print(f"\nDisplaying first {min(num_print, total_errors)} misclassified samples:\n")
        
        header = f"{'True Label':<22} | {'1st Guess (Wrong)':<28} | {'2nd Guess':<28} | {'3rd Guess':<28}"
        print(header)
        print("-" * len(header))

        for i in range(min(num_print, total_errors)):
            true_id = y_true_errors[i]
            row_ids = top3_ids_errors[i]
            row_probs = top3_probs_errors[i]
            
            # Helper to get bird name
            def get_name(cid):
                if class_names:
                    try:
                        name = class_names[cid] if isinstance(class_names, dict) else class_names[cid]
                        return (name[:18] + '..') if len(name) > 18 else name
                    except: pass
                return str(cid)

            # Helper to format "Name (0.95)"
            def fmt(cid, prob):
                name = get_name(cid)
                marker = "Correct" if cid == true_id else "" 
                return f"{marker}{name} ({prob:.2f})"

            # Build row
            true_name = get_name(true_id)
            
            col1 = fmt(row_ids[0], row_probs[0]) # This will never have checkmark (it's the error)
            col2 = fmt(row_ids[1], row_probs[1]) # Checkmark might appear here
            col3 = fmt(row_ids[2], row_probs[2]) # Checkmark might appear here
            
            print(f"{true_name:<22} | {col1:<28} | {col2:<28} | {col3:<28}")

        # 6. SUMMARY STATISTICS FOR ERRORS
        print("\n" + "-"*90)
        
        # Check how often the correct label was "almost" chosen (Rank 2 or 3)
        correct_at_2 = np.mean(top3_ids_errors[:, 1] == y_true_errors)
        correct_at_3 = np.mean(top3_ids_errors[:, 2] == y_true_errors)
        missed_entirely = 1.0 - correct_at_2 - correct_at_3
        
        print(f"Error Analysis:")
        print(f"- Correct label was 2nd choice: {correct_at_2:.1%}")
        print(f"- Correct label was 3rd choice: {correct_at_3:.1%}")
        print(f"- Correct label NOT in Top 3:   {missed_entirely:.1%} (Severe misclassification)")
        print("="*90)

def diagnose_high_overlap_drop(df, target_indices, model_id2label, class_names=None):
    # 1. Filter for Valid Target Data
    df_valid = df[df['true_id'].isin(target_indices)].copy()
    
    # 2. Re-calculate Correctness (Closed-Set)
    full_probs = np.stack(df_valid['probs'].values)
    subset_probs = full_probs[:, target_indices]
    relative_preds = np.argmax(subset_probs, axis=1)
    target_arr = np.array(target_indices)
    y_pred_closed = target_arr[relative_preds]
    
    df_valid['is_correct'] = (df_valid['true_id'] == y_pred_closed).astype(int)
    df_valid['pred_id_closed'] = y_pred_closed # Save for printing

    # 3. Create Bins
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    df_valid['overlap_bin'] = pd.cut(df_valid['overlap'], bins=bins, labels=labels, include_lowest=True)

    # 4. Global Stats per Bin
    print("="*60)
    print(f"{'Bin':<10} | {'Count':<8} | {'Accuracy':<10} | {'Errors':<8}")
    print("-" * 60)
    
    bin_stats = df_valid.groupby('overlap_bin', observed=False).agg(
        count=('is_correct', 'size'),
        accuracy=('is_correct', 'mean'),
        errors=('is_correct', lambda x: (1-x).sum())
    )
    
    for bin_label, row in bin_stats.iterrows():
        print(f"{bin_label:<10} | {int(row['count']):<8} | {row['accuracy']:.2%}    | {int(row['errors']):<8}")
    print("="*60)

    # 5. DEEP DIVE: Inspect the 80-100% Bin
    high_overlap_df = df_valid[df_valid['overlap_bin'] == '80-100%']
    
    if len(high_overlap_df) == 0:
        print("No samples in the 80-100% bin to analyze.")
        return

    # A. Check Species Distribution in this bin
    print("\nSpecies composition of the 80-100% bin:")
    species_counts = high_overlap_df['true_id'].value_counts()
    
    for class_id, count in species_counts.items():
        # Calculate accuracy for this specific species within this bin
        spec_df = high_overlap_df[high_overlap_df['true_id'] == class_id]
        acc = spec_df['is_correct'].mean()
        
        # Get Name
        name = str(class_id)
        class_name = model_id2label[class_id]
        if class_names:
             name = class_names.get(class_id, str(class_id))
        
        print(f"- {name:<20} ({class_name}): {count} samples, Accuracy: {acc:.1%}")

    # B. List the Specific Errors
    errors_df = high_overlap_df[high_overlap_df['is_correct'] == 0]
    
    print(f"\nFound {len(errors_df)} specific errors in the 80-100% bin:")
    print("-" * 80)
    print(f"{'Filename':<30} | {'True Label':<20} | {'Predicted':<20} | {'Conf'}")
    print("-" * 80)
    
    for idx, row in errors_df.iterrows():
        fname = row.get('fullfilename', 'Unknown_File') # Adjust column name if needed
        true_name = str(row['true_id'])
        pred_name = str(row['pred_id_closed'])
        
        if class_names:
            true_name = class_names.get(row['true_id'], true_name)
            pred_name = class_names.get(row['pred_id_closed'], pred_name)
            
        # Get confidence of the prediction
        # (Using the probability of the predicted class)
        pred_idx = list(target_indices).index(row['pred_id_closed'])
        conf = row['probs'][row['pred_id_closed']] # Or handle if probs is list
        
        print(f"{fname[:30]:<30} | {true_name:<20} | {pred_name:<20} | {conf:.2f}")

def analyze_hallucinations(df):
    print("\n" + "="*50)
    print("HALLUCINATION ANALYSIS (Predictions on NA Inputs)")
    print("="*50)
    
    # 1. Filter ONLY the NA samples
    df_na = df[df['true_id'] == -1].copy()
    
    if len(df_na) == 0:
        print("No NA samples found to analyze.")
        return

    # 2. Group by what the model PREDICTED
    stats = df_na.groupby('pred_str')['confidence'].agg(['count', 'mean', 'max']).reset_index()
    
    # Sort by 'count' (Most frequent hallucinations) or 'mean' (Strongest hallucinations)
    stats = stats.sort_values('mean', ascending=False)
    
    print(f"{'Predicted Species':<25} {'Count':<8} {'Avg Conf':<10} {'Max Conf':<10}")
    print("-" * 55)
    
    for _, row in stats.iterrows():
        print(f"{row['pred_str'][:23]:<25} {int(row['count']):<8} {row['mean']:.4f}     {row['max']:.4f}")
        
    # 3. PLOT: Top 10 Strongest Hallucinations
    valid_stats = stats[stats['count'] >= 5]
    if len(valid_stats) > 0:
        top_hallucinations = valid_stats.head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_hallucinations, x='mean', y='pred_str', palette='magma')
        plt.title("Strongest Hallucinations: Average Confidence when Input is NA")
        plt.xlabel("Average Confidence")
        plt.ylabel("Predicted Species")
        plt.xlim(0, 1.0)
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("graph_hallucination_strength.png")
        print("\nSaved graph_hallucination_strength.png")
    else:
        print("\nNot enough repeated hallucinations to generate a graph.")

if __name__ == "__main__":
    results_df, target_indices, id2label = run_evaluation()
    analyze_results(results_df, target_indices)
    diagnose_high_overlap_drop(results_df, target_indices, id2label)
    analyze_errors_topk(results_df, target_indices)
    analyze_hallucinations(results_df)
    results_df.drop(columns=['probs']).to_csv("final_evaluation_results1.csv", index=False)