import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import librosa
import json
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoFeatureExtractor
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from tqdm import tqdm
from datasets import load_dataset, Audio
from sklearn.preprocessing import MultiLabelBinarizer

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    # Paths
    "csv_path": "/home/users/ntu/ytong005/VAD/javad/JavadPreds_top20_for_classification.csv",
    "dataset_json": "dataset_json/per_ebird.json",
    "base_model": "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
    "save_dir": "./finetuned_models",
    
    # Parameters
    "sample_rate": 32000,
    "chunk_batch_size": 16,  # Inference memory safety
    "batch_size": 4,         # Training batch size
    "epochs": 10,            # Finetuning epochs
    "lr_head": 1e-3,         # Learning rate for new prototypes
    "lr_backbone": 1e-6,     # Very slow learning for backbone
    
    # Prototype Settings
    "base_k": 5,             # Original model has 5 protos per class
    "hard_k": 10,            # We want 10 for hard classes
    "acc_threshold": 0.40,   # Classes below 50% accuracy are "Hard"
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==========================================
# 2. UTILS & DATASET
# ==========================================
# model id2label is str:str
def build_label_mapping(dataset_json, model_label2id):
    """Maps Dataset IDs to Model IDs, filtering only for birds in the CSV."""
    with open(dataset_json, 'r') as f:
        ds_data = json.load(f)
        ds_id2label = {int(k): v for k, v in ds_data['id2label'].items()} # int:str

    ds_to_original_model = {} # int:int
    target_indices = [] # original model id, int
    ignored_indices = []

    for ds_id, ds_label in ds_id2label.items():
        if ds_label in model_label2id:
            model_id = model_label2id[ds_label] # str
            ds_to_original_model[ds_id] = int(model_id)
            target_indices.append(int(model_id)) # list of ints
        else:
            print(f"⚠️ Warning: Dataset class '{ds_label}' (ID {ds_id}) not found in Model. Ignoring.")
            ignored_indices.append(ds_id)

    print(f"Ignored {len(ignored_indices)} dataset classes not in model")
    return ds_to_original_model, sorted(list(set(target_indices)))

def inference_collate_fn(batch):
    """Custom collate to preserve label structure"""
    return {
        'audio': [item['audio'] for item in batch],
        'label_id': [item['label_id'] for item in batch]
    }

class InferenceDataset(Dataset):
    def __init__(self, df, id_map, train_mode=False):
        self.df = df
        self.sr = CONFIG["sample_rate"]
        self.id_map = id_map
        self.target_len = self.sr * 5
        self.train_mode = train_mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df[idx]
        path = row['audio']["path"]
        raw_cat = row['labels']
        
        # Resolve Label
        label_id = []
        try:
            for i in raw_cat:
                if int(i) in self.id_map: 
                    label_id.append(self.id_map[int(i)])  # returning original model_id
        except: 
            pass

        chunks = []
        try:
            y, _ = librosa.load(path, sr=self.sr, mono=True)
            if len(y) > 0:
                if self.train_mode:
                    # TRAINING: Random Crop or Loop
                    if len(y) < self.target_len:
                        n_repeats = int(np.ceil(self.target_len / len(y)))
                        y = np.tile(y, n_repeats)[:self.target_len]
                    elif len(y) > self.target_len:
                        start = np.random.randint(0, len(y) - self.target_len)
                        y = y[start:start+self.target_len]
                    chunks.append(y)
                else:
                    # INFERENCE: Sliding Window
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
        except Exception as e:
            chunks.append(np.zeros(self.target_len, dtype=np.float32))

        return {"audio": chunks, "label_id": label_id}

def run_inference(model, feature_extractor, dataset, target_indices):
    """
    Universal Inference Function.
    Works for both Full Model (9700 classes) and Reduced Model (132 classes).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, 
                       collate_fn=inference_collate_fn)
    
    # MASTER MAP: Original_Model_ID (e.g. 7988) -> New_Evaluation_Index (e.g. 5)
    old2new = {oid: i for i, oid in enumerate(target_indices)} # int:int
    
    print(f"\n🔍 Inference Setup:")
    print(f"  Target Classes: {len(target_indices)}")
    print(f"  Model Output Dimension: {model.config.num_labels}")
    
    # Detection: Is the model already reduced?
    # We check if the output dimension matches our target list length
    is_reduced_model = (model.config.num_labels == len(target_indices))
    if is_reduced_model:
        print("  --> Detected REDUCED Model (Post-Surgery)")
    else:
        print("  --> Detected FULL Model (Pre-Surgery)")

    y_true_indices = [] # Will store NEW indices [0..131]
    y_probs_all = []    # Will store vectors of size [132]
    
    print(f"Running Inference on {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            raw_chunks = batch['audio'][0]
            label_ids = batch['label_id'][0] # List of Original IDs [7988, 8000]
            
            # 1. RESOLVE TRUTH TO NEW INDICES [0..131]
            # ---------------------------------------------------
            sample_true_indices = []
            for l in label_ids:
                if l in old2new:
                    sample_true_indices.append(old2new[l]) # Maps 7988 -> 5
            
            # If this recording contains no birds from our target list, skip
            if not sample_true_indices:
                continue

            # 2. RUN MODEL
            # ---------------------------------------------------
            file_probs = []
            for i in range(0, len(raw_chunks), CONFIG["chunk_batch_size"]):
                mb = raw_chunks[i : i + CONFIG["chunk_batch_size"]]
                inputs = feature_extractor(mb, padding=True, return_tensors="pt")
                inputs = inputs.to(CONFIG["device"])
                
                outputs = model(inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu()
                file_probs.append(probs)

            if not file_probs: continue

            # Max Voting across time chunks
            final_probs_full, _ = torch.max(torch.cat(file_probs), dim=0)
            
            # 3. NORMALIZE PREDICTIONS TO NEW INDICES [0..131]
            # ---------------------------------------------------
            if is_reduced_model:
                subset_probs = final_probs_full.numpy()
            else:
                subset_probs = final_probs_full[target_indices].numpy()
                
                # Renormalize so they sum to 1 (optional but cleaner)
                subset_probs = subset_probs / (np.sum(subset_probs) + 1e-12)

            y_true_indices.append(sample_true_indices)
            y_probs_all.append(subset_probs)

    if len(y_true_indices) == 0:
        print("⚠️ No valid samples found!")
        return 0.0, 0.0, 0.0, 0.0, {}

    # ==========================================
    # METRICS
    # ==========================================
    y_score = np.array(y_probs_all)

    mlb = MultiLabelBinarizer(classes=range(len(target_indices)))
    y_true_binary = mlb.fit_transform(y_true_indices)

    print("\nCalculating Metrics...")
    
    # Top-1 Accuracy
    top1_preds = np.argmax(y_score, axis=1)
    correct_mask = y_true_binary[np.arange(len(y_true_binary)), top1_preds] == 1
    top1_acc = np.mean(correct_mask)

    # Top-3 Accuracy
    k = min(3, len(target_indices))
    topk_preds = np.argsort(y_score, axis=1)[:, -k:][:, ::-1]
    topk_hits = 0
    for i in range(len(y_true_binary)):
        if not set(y_true_indices[i]).isdisjoint(topk_preds[i]):
            topk_hits += 1
    top3_acc = topk_hits / len(y_true_binary)

    # cmAP & AUROC
    try:
        cmAP = average_precision_score(y_true_binary, y_score, average='macro')
        AUROC = roc_auc_score(y_true_binary, y_score, multi_class='ovr', average='macro')
    except:
        cmAP, AUROC = 0.0, 0.0
        
    # Per-Class Accuracy
    class_acc = {}
    for i, oid in enumerate(target_indices):
        relevant_samples = (y_true_binary[:, i] == 1)
        if np.sum(relevant_samples) > 0:
            matches = (top1_preds[relevant_samples] == i)
            class_acc[oid] = np.mean(matches)
        else:
            class_acc[oid] = 0.0

    print(f"✅ Top-1: {top1_acc:.4f} | Top-3: {top3_acc:.4f} | cmAP: {cmAP:.4f} | AUROC: {AUROC:.4f}")
    
    return top1_acc, top3_acc, cmAP, AUROC, class_acc

def perform_weight_surgery(model, target_indices):
    """Slices the classification head to keep only target classes."""
    print(f"\n{'='*60}")
    print(f"PERFORMING WEIGHT SURGERY")
    print(f"{'='*60}")
    
    with torch.no_grad():
        # Get old prototype count
        old_num_protos = model.head.prototype_vectors.shape[0]
        
        # Calculate new prototype count
        k = CONFIG["base_k"]  # 5
        indices_to_keep = []
        for idx in target_indices:
            start = idx * k
            indices_to_keep.extend(range(start, start + k))
        
        new_num_protos = len(indices_to_keep)
        new_num_classes = len(target_indices)
        
        print(f"\n📊 Surgery Plan:")
        print(f"  Prototypes: {old_num_protos} -> {new_num_protos}")
        print(f"  Classes: {model.config.num_labels} -> {new_num_classes}")
        
        # 1. Slice prototype_vectors
        print(f"\n Slicing prototype_vectors...")
        old_protos = model.head.prototype_vectors.data
        model.head.prototype_vectors = nn.Parameter(old_protos[indices_to_keep])
        print(f"  ✓ New shape: {model.head.prototype_vectors.shape}")
        
        # 2. Slice all buffers with first dimension == old_num_protos
        print(f"\n Slicing prototype-related buffers:")
        buffers_to_update = {}
        for name, buffer in list(model.head.named_buffers()):
            if buffer.shape[0] == old_num_protos:
                print(f"  - {name}: {buffer.shape} -> ", end="")
                buffers_to_update[name] = buffer[indices_to_keep]
                print(f"{buffers_to_update[name].shape}")
        
        # Update buffers
        for name, new_buffer in buffers_to_update.items():
            delattr(model.head, name)
            model.head.register_buffer(name, new_buffer)
        
        # 3. Create new Identity Matrix
        print(f"\n Creating new prototype_class_identity...")
        new_identity = torch.zeros(new_num_protos, new_num_classes, device=model.device)
        for i in range(new_num_classes):
            new_identity[i*k : (i+1)*k, i] = 1.0
        
        if hasattr(model.head, 'prototype_class_identity'):
            delattr(model.head, 'prototype_class_identity')
        model.head.register_buffer('prototype_class_identity', new_identity)
        print(f"  ✓ Shape: {new_identity.shape}")
        
        # 4. Rebuild last_layer
        print(f"\n Rebuilding last_layer...")
        print(f"  Old: [{model.head.last_layer.in_features} x {model.head.last_layer.out_features}]")
        
        model.head.last_layer = nn.Linear(new_num_protos, new_num_classes, bias=False)
        model.head.last_layer.weight.data.copy_(new_identity.t())
        model.head.last_layer.weight.requires_grad = False
        
        print(f"  New: [{model.head.last_layer.in_features} x {model.head.last_layer.out_features}]")
        
        # 5. Update metadata
        print(f"\n📝 Updating model metadata...")
        model.config.num_labels = new_num_classes
        if hasattr(model.head, 'num_classes'):
            model.head.num_classes = new_num_classes
        if hasattr(model.head, 'num_prototypes'):
            model.head.num_prototypes = new_num_protos
        if hasattr(model.head, 'prototype_shape'):
            old_shape = model.head.prototype_shape
            new_shape = (new_num_protos,) + old_shape[1:]
            model.head.prototype_shape = new_shape
            print(f"  prototype_shape: {old_shape} -> {new_shape}")

        # Create new mapping
        print(f"\n Creating new label mappings...")
        old_id2label = model.config.id2label.copy()
        new_id2label = {}
        new_label2id = {}
        
        for new_id, old_id in enumerate(target_indices):
            label_name = old_id2label[str(old_id)]
            new_id2label[new_id] = label_name
            new_label2id[label_name] = new_id
        
        model.config.id2label = new_id2label
        model.config.label2id = new_label2id
        print(f" Updated id2label: {len(new_id2label)} classes")
        
    print(f"\n Surgery Complete!")
    print(f"{'='*60}\n")
    return model.to(CONFIG["device"])

def perform_prototype_surgery(model, target_indices, hard_classes, old_id2label):
    """
    Increases prototypes from 5 to 10 for hard classes.
    """
    print(f"\n{'='*60}")
    print(f"PERFORMING DYNAMIC PROTOTYPE SURGERY")
    print(f"{'='*60}")
    
    device = model.device
    base_k = CONFIG["base_k"]  # 5
    
    # 1. Define New Counts
    print(f"\n📊 Prototype Allocation:")
    class_counts = {}
    for i, original_id in enumerate(target_indices):
        is_hard = original_id in hard_classes
        k = CONFIG["hard_k"] if is_hard else base_k
        class_counts[i] = k
        name = old_id2label[str(original_id)]
        status = "HARD" if is_hard else "Normal"
        print(f"  {name:<30} : {k:2d} protos ({status})")
        
    # 2. Calculate totals
    total_new = sum(class_counts.values())
    c, h, w = model.head.prototype_vectors.shape[1:]
    old_num_protos = model.head.prototype_vectors.shape[0]
    
    print(f"\n Surgery Plan:")
    print(f"  Total prototypes: {old_num_protos} -> {total_new}")
    print(f"  Hard classes: {len(hard_classes)}")
    
    # 3. Create new tensors
    new_protos = torch.zeros((total_new, c, h, w), device=device)
    new_identity = torch.zeros((total_new, len(target_indices)), device=device)
    
    # 4. Prepare to expand buffers
    buffers_to_expand = {}
    for name, buffer in model.head.named_buffers():
        if buffer.shape[0] == old_num_protos and name != 'prototype_class_identity':
            new_shape = list(buffer.shape)
            new_shape[0] = total_new
            buffers_to_expand[name] = torch.zeros(new_shape, device=device)
    
    # 5. Clone
    print(f"\n Cloning prototypes...")
    old_protos = model.head.prototype_vectors.data
    current_idx = 0
    
    for i in range(len(target_indices)):
        target_k = class_counts[i]
        
        # Get old prototypes for this class
        old_start = i * base_k
        old_block = old_protos[old_start : old_start + base_k]
        
        # Copy original prototypes
        new_protos[current_idx : current_idx + base_k] = old_block
        
        # Copy buffers
        for name, new_buffer in buffers_to_expand.items():
            old_buffer = dict(model.head.named_buffers())[name]
            old_buffer_block = old_buffer[old_start : old_start + base_k]
            new_buffer[current_idx : current_idx + base_k] = old_buffer_block
        
        # Clone additional prototypes for hard classes
        if target_k > base_k:
            needed = target_k - base_k
            for n in range(needed):
                clone = old_block[n % base_k].clone()
                noise = torch.randn_like(clone) * 0.02
                new_protos[current_idx + base_k + n] = clone + noise
                
                # Clone buffers
                for name, new_buffer in buffers_to_expand.items():
                    old_buffer = dict(model.head.named_buffers())[name]
                    old_buffer_block = old_buffer[old_start : old_start + base_k]
                    new_buffer[current_idx + base_k + n] = old_buffer_block[n % base_k].clone()
        
        # Set Identity
        new_identity[current_idx : current_idx + target_k, i] = 1.0
        current_idx += target_k

    # 6. Update Model
    print(f"\n Updating model...")
    model.head.prototype_vectors = nn.Parameter(new_protos)
    
    # Update buffers
    for name, new_buffer in buffers_to_expand.items():
        delattr(model.head, name)
        model.head.register_buffer(name, new_buffer)
        print(f"  Buffer '{name}': {new_buffer.shape}")
    
    # Update identity
    if hasattr(model.head, 'prototype_class_identity'):
        delattr(model.head, 'prototype_class_identity')
    model.head.register_buffer('prototype_class_identity', new_identity)
    print(f"  prototype_class_identity: {new_identity.shape}")
        
    # 7. Rebuild Last Layer
    model.head.last_layer = nn.Linear(total_new, len(target_indices), bias=False)
    model.head.last_layer.weight.data.copy_(new_identity.t())
    model.head.last_layer.weight.requires_grad = False
    print(f"  last_layer: [{total_new} x {len(target_indices)}]")
    
    # 8. Update counts
    if hasattr(model.head, 'num_prototypes'):
        model.head.num_prototypes = total_new
    if hasattr(model.head, 'num_classes'):
        model.head.num_classes = len(target_indices)
    if hasattr(model.head, 'prototype_shape'):
        old_shape = model.head.prototype_shape
        new_shape = (total_new,) + old_shape[1:]
        model.head.prototype_shape = new_shape
    
    print(f"\n Dynamic Surgery Complete!")
    print(f"{'='*60}\n")
    return model.to(device)

def finetune_model(model, train_df, ds_to_original_model, target_indices, feature_extractor):
    print(f"\n{'='*60}")
    print(f"FINE-TUNING MODEL")
    print(f"{'='*60}")
    
    # Freeze Backbone, train only head
    trainable_params = []
    for name, param in model.named_parameters():
        if "head" in name and "prototype_vectors" in name:
            param.requires_grad = True
            trainable_params.append(param)
            print(f"✓ Training: {name}")
        else:
            param.requires_grad = False
    
    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=CONFIG["lr_head"])
    criterion = nn.CrossEntropyLoss()
    old2new = {oid: i for i, oid in enumerate(target_indices)}
    
    def train_collate(batch):
        """Collate function for training"""
        audio_arrays = []
        valid_labels = []
        
        for item in batch:
            audio_chunks = item["audio"]
            label_ids = item["label_id"]
            
            if not label_ids:
                continue
            
            # Map to new indices
            new_labels = [old2new[l] for l in label_ids if l in old2new]
            if not new_labels:
                continue
            
            # Use first chunk for training
            if len(audio_chunks) > 0:
                audio_arrays.append(audio_chunks[0])
                # For multi-label, we'll use the first label
                valid_labels.append(new_labels[0])
        
        if len(audio_arrays) == 0:
            return None, None
        
        inputs = feature_extractor(audio_arrays, padding=True, return_tensors="pt", 
                                   sampling_rate=CONFIG["sample_rate"])
        labels = torch.tensor(valid_labels, dtype=torch.long)
        
        return inputs, labels

    # Create Train/Val Loaders
    train_dataset, val_dataset = train_df.train_test_split(test_size=0.2, seed=42)
    train_ds = InferenceDataset(train_dataset, ds_to_original_model, train_mode=True)
    val_ds = InferenceDataset(val_dataset, ds_to_original_model, train_mode=True)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, 
                             collate_fn=train_collate, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, 
                           collate_fn=train_collate, num_workers=4)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"\n Starting training for {CONFIG['epochs']} epochs...")
    
    for epoch in range(CONFIG["epochs"]):
        # TRAIN
        model.train()
        running_train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for batch_data in pbar:
            if batch_data[0] is None:
                continue
                
            inputs, labels = batch_data
            inputs = inputs.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix(loss=loss.item())
        
        if train_batches == 0:
            print(" No valid training batches!")
            continue
            
        avg_train_loss = running_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # VALIDATE
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data[0] is None:
                    continue
                    
                inputs, labels = batch_data
                inputs = inputs.to(CONFIG["device"])
                labels = labels.to(CONFIG["device"])
                
                outputs = model(inputs)
                loss = criterion(outputs.logits, labels)
                running_val_loss += loss.item()
                val_batches += 1
        
        if val_batches == 0:
            print(" No valid validation batches!")
            continue
                
        avg_val_loss = running_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "model_dynamic_best.pth"))
            print("   Best model saved!")
    
    # Plot training curve
    if len(train_losses) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training vs Validation Loss (Dynamic)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(CONFIG["save_dir"], "dynamic_training_curve.png"))
        print(f"\n Training curve saved!")

    print(f"\n Fine-tuning complete!")
    print(f"{'='*60}\n")
    return model

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    print("\n" + "="*60)
    print("AUDIOPROTONET DYNAMIC PROTOTYPE SURGERY PIPELINE")
    print("="*60 + "\n")
    
    # 1. Load Data & Base Model
    print(" Loading dataset...")
    dataset = load_dataset(
        name="PER",
        path='DBD-research-group/BirdSet',
        trust_remote_code=True,
        cache_dir='/home/users/ntu/ytong005/scratch/Birdset_PER'
    )
    dataset = dataset.rename_column("ebird_code_multilabel", "labels")
    dataset = dataset.select_columns(["audio", "labels"])
    
    print(" Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model"], 
        trust_remote_code=True
    )
    model.to(CONFIG["device"])

    original_id2label = model.config.id2label.copy()
    
    print(" Building label mapping...")
    ds_map, target_indices = build_label_mapping(
        CONFIG["dataset_json"], 
        model.config.label2id
    )
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        CONFIG["base_model"], 
        trust_remote_code=True
    )
    
    print(f"\n✓ Target classes: {len(target_indices)}")
    print(f"✓ Dataset mapping: {len(ds_map)} entries")
    
    # Create test dataset
    test_ds = InferenceDataset(dataset["test_5s"], ds_map, train_mode=False)
    print(f"✓ Test samples: {len(test_ds)}")

    # ---------------------------------------------------------
    # STEP 1: BASELINE INFERENCE (Original Model)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: BASELINE INFERENCE (Original Model)")
    print("="*60)
    
    base_top1_acc, base_top3_acc, base_cmAP, base_auroc, base_class_acc = \
        run_inference(model, feature_extractor, test_ds, target_indices)
    
    # Identify hard classes
    hard_classes = []
    print(f"\n Per-Class Accuracy (Threshold: {CONFIG['acc_threshold']:.0%}):")
    print(f"{'Class':<30} {'Accuracy':<10} {'Status'}")
    print("-" * 55)
    for oid in target_indices:
        acc = base_class_acc[oid]
        name = model.config.id2label[str(oid)]
        status = " Good" if acc >= CONFIG["acc_threshold"] else " HARD"
        print(f"{name:<30} {acc:>6.2%}     {status}")
        if acc < CONFIG["acc_threshold"]:
            hard_classes.append(oid)
    
    print(f"\n Summary: {len(hard_classes)} hard classes found")

   

    # ---------------------------------------------------------
    # STEP 2: WEIGHT SURGERY (Reduce Model)
    # ---------------------------------------------------------
    model = perform_weight_surgery(model, target_indices)
    
    # ---------------------------------------------------------
    # STEP 3: POST-SURGERY EVALUATION
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: POST-SURGERY EVALUATION")
    print("="*60)
    
    reduced_top1_acc, reduced_top3_acc, reduced_cmAP, reduced_auroc, reduced_class_acc = \
        run_inference(model, feature_extractor, test_ds, target_indices)

    # ---------------------------------------------------------
    # STEP 4: DYNAMIC PROTOTYPE SURGERY
    # ---------------------------------------------------------
    model = perform_prototype_surgery(model, target_indices, hard_classes, original_id2label)

    # ---------------------------------------------------------
    # STEP 5: POST-DYNAMIC SURGERY EVALUATION
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: POST-DYNAMIC SURGERY EVALUATION")
    print("="*60)
    
    dynamic_top1_acc, dynamic_top3_acc, dynamic_cmAP, dynamic_auroc, dynamic_class_acc = \
        run_inference(model, feature_extractor, test_ds, target_indices)

    # ---------------------------------------------------------
    # STEP 6: FINE-TUNING
    # ---------------------------------------------------------
    model = finetune_model(model, dataset["train"], ds_map, target_indices, feature_extractor)
    
    # ---------------------------------------------------------
    # STEP 7: FINAL EVALUATION
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4: FINAL EVALUATION (After Fine-tuning)")
    print("="*60)
    
    final_top1_acc, final_top3_acc, final_cmAP, final_auroc, final_class_acc = \
        run_inference(model, feature_extractor, test_ds, target_indices)
    
    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Metric':<25} {'Baseline':<12} {'Reduced':<12} {'Dynamic':<12} {'Finetuned'}")
    print("-" * 73)
    print(f"{'Top-1 Accuracy':<25} {base_top1_acc:>10.4f}  {reduced_top1_acc:>10.4f}  {dynamic_top1_acc:>10.4f}  {final_top1_acc:>10.4f}")
    print(f"{'Top-3 Accuracy':<25} {base_top3_acc:>10.4f}  {reduced_top3_acc:>10.4f}  {dynamic_top3_acc:>10.4f}  {final_top3_acc:>10.4f}")
    print(f"{'cmAP':<25} {base_cmAP:>10.4f}  {reduced_cmAP:>10.4f}  {dynamic_cmAP:>10.4f}  {final_cmAP:>10.4f}")
    print(f"{'AUROC':<25} {base_auroc:>10.4f}  {reduced_auroc:>10.4f}  {dynamic_auroc:>10.4f}  {final_auroc:>10.4f}")
    print("="*60)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "dynamic_proto_model_final.pth"))
    print(f"\n Final model saved to {CONFIG['save_dir']}")
    print("\n Pipeline Complete!\n")

if __name__ == "__main__":
    main()