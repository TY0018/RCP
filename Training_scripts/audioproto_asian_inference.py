import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
import os
import librosa
# ==========================================
# 1. SETUP & MAPPING
# ==========================================
variant = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_CSV_PATH = "/home/users/ntu/ytong005/scratch/asian_bird_dataset/Asian_Birds_Split/test.csv"

SPECIES_MAPPING = {
    "Acridotheres_tristis": "commyn",
    "Aethopyga_siparaja": "eacsun1",
    "Amaurornis_phoenicurus": "whbwat1",
    "Anthracoceros_albirostris": "orphor1",
    "Aplonis_panayensis": "asgsta1",
    "Columba_livia": "rocpig",
    "Copsychus_saularis": "magrob",
    "Corvus_splendens": "houcro1",
    "Dicaeum_cruentatum": "scbflo1",
    "Gallus_gallus": "redjun",
    "Hirundo_tahitica": "pacswa1",
    "Oriolus_chinensis": "blnori1",
    "Orthotomus_sutorius": "comtai1",
    "Passer_montanus": "eutspa",
    "Pycnonotus_jocosus": "rewbul",
    "Todiramphus_chloris": "colkin1",
    "Treron_vernans": "pinpig3",
    "Acridotheres javanicus": "whvmyn",
    "Gracula religiosa": "hilmyn"
}

# Load Model
model_name = f"DBD-research-group/AudioProtoPNet-{variant}-BirdSet-XCL"
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
model.eval().to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)

# ==========================================
# 2. LOAD LOCAL DATASET
# ==========================================
print("Loading local dataset...")
df_test = pd.read_csv(TEST_CSV_PATH)

first_file = df_test['fullfilename'].iloc[0]
if not os.path.exists(first_file):
    print(f"⚠️  WARNING: Could not find file: {first_file}")

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df_test)

# ==========================================
# 3. PREPARE CLASSES
# ==========================================
# Get the model's internal label map (eBird Code -> ID)
model_label2id = model.config.label2id
model_id2label = model.config.id2label

valid_class_indices = []
print("\nChecking Class Overlap:")
for sci_name, ebird_code in SPECIES_MAPPING.items():
    if ebird_code in model_label2id:
        idx = model_label2id[ebird_code]
        valid_class_indices.append(idx)
        print(f"✅ Found: {sci_name} -> {ebird_code} (ID: {idx})")
    else:
        print(f"⚠️  MISSING in Model: {sci_name} mapped to {ebird_code}")

if not valid_class_indices:
    raise ValueError("No overlapping classes found! Check SPECIES_MAPPING eBird codes.")

# ==========================================
# 4. RUN INFERENCE
# ==========================================
all_probs = []
all_targets = [] 

print(f"\nEvaluating on {len(dataset)} files...")

for sample in tqdm(dataset):
    
    path = sample["fullfilename"]
    
    try:
        audio_array, _ = librosa.load(path, sr=32000, mono=True)
    except Exception as e:
        print(f"⚠️ Error loading {path}: {e}")
        continue # Skip corrupted files

    # Check for empty/short audio
    if len(audio_array) == 0: continue

    true_species = sample["categories"] 
    
    # Skip if we don't have a mapping for this file's species
    if true_species not in SPECIES_MAPPING:
        continue
        
    mapped_ebird_code = SPECIES_MAPPING[true_species]
    
    # C. Model Prediction
    with torch.no_grad():
        # Preprocess
        inputs = feature_extractor(audio_array)
        inputs = inputs.to(device)
        
        # Forward pass
        output = model(inputs)
        logits = output.logits
        
        # Mask irrelevant classes
        mask = torch.full_like(logits, float('-inf'))
        mask[:, valid_class_indices] = logits[:, valid_class_indices]
        
        probs = torch.sigmoid(mask).cpu().numpy().squeeze()
        all_probs.append(probs)

    # D. Create Target Vector
    target = np.zeros(len(model_label2id))
    if mapped_ebird_code in model_label2id:
        true_idx = model_label2id[mapped_ebird_code]
        target[true_idx] = 1.0
    all_targets.append(target)

if len(all_probs) == 0:
    print("\n🔴 CRITICAL ERROR: No audio files were processed!")
    print("This means 'librosa.load' failed for every single file.")
    exit() 
    
# ==========================================
# 5. CALCULATE METRICS
# ==========================================
all_probs = np.array(all_probs)
all_targets = np.array(all_targets)

print("\nCalculating metrics...")

# cmAP (Class-mean Average Precision)
try:
    ap_scores = []
    for i in valid_class_indices:
        # Only calculate if this class actually exists in the test set targets
        if np.sum(all_targets[:, i]) > 0:
            ap = average_precision_score(all_targets[:, i], all_probs[:, i])
            ap_scores.append(ap)
    cmAP = np.mean(ap_scores)
except Exception as e:
    print(f"Error calc cmAP: {e}")
    cmAP = 0.0

# AUROC = mean per-class ROC AUC
try:
    AUROC = np.mean([
        roc_auc_score(all_targets[:, i], all_probs[:, i])
        for i in valid_class_indices
        if len(np.unique(all_targets[:, i])) > 1
    ])
except ValueError:
    AUROC = float("nan")

# Top-1 Accuracy
pred_indices = np.argmax(all_probs, axis=1) # The class with highest prob
true_indices = np.argmax(all_targets, axis=1) # The class with 1.0
correct = (pred_indices == true_indices)
t1_acc = np.mean(correct)

print("------------------------------------------------")
print(f"Results on Custom Dataset ({len(all_probs)} samples)")
print("------------------------------------------------")
print(f"cmAP:    {cmAP:.4f}")
print(f"AUROC: {AUROC:.4f}")
print(f"T1-Acc:  {t1_acc:.4f}")
print("------------------------------------------------")