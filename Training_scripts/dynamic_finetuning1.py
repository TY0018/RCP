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
    "acc_threshold": 0.40,   # Classes below 40% accuracy are "Hard"
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==========================================
# 2. UTILS & DATASET
# ==========================================
def build_label_mapping(dataset_json, model_label2id):
    """
    Maps Dataset IDs to Original Model IDs via label names.
    
    Returns:
        ds_to_original_model: dict {dataset_id: original_model_id}
        target_indices: sorted list of original model IDs present in dataset
    """
    with open(dataset_json, 'r') as f:
        ds_data = json.load(f)
        ds_id2label = {int(k): v for k, v in ds_data['id2label'].items()}

    ds_to_original_model = {}
    target_indices = []
    ignored_indices = []

    for ds_id, ds_label in ds_id2label.items():
        if ds_label in model_label2id:
            original_model_id = int(model_label2id[ds_label])
            ds_to_original_model[ds_id] = original_model_id
            target_indices.append(original_model_id)
        else:
            print(f" Dataset class '{ds_label}' (ID {ds_id}) not in model")
            ignored_indices.append(ds_id)

    print(f" Found {len(ds_to_original_model)} matching classes")
    print(f" Ignored {len(ignored_indices)} dataset classes not in model")
    return ds_to_original_model, sorted(list(set(target_indices)))

def inference_collate_fn(batch):
    """Custom collate to preserve label structure"""
    return {
        'audio': [item['audio'] for item in batch],
        'label_id': [item['label_id'] for item in batch]
    }

class InferenceDataset(Dataset):
    def __init__(self, df, ds_to_original_model, train_mode=False):
        """
        Args:
            df: HuggingFace dataset split
            ds_to_original_model: dict mapping dataset_id -> original_model_id
            train_mode: if True, use random crops; if False, use sliding windows
        """
        self.df = df
        self.sr = CONFIG["sample_rate"]
        self.ds_to_original_model = ds_to_original_model
        self.target_len = self.sr * 5
        self.train_mode = train_mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df[idx]
            
            # Handle different audio field formats from HuggingFace datasets
            audio_field = row['audio']
            if isinstance(audio_field, dict):
                if 'path' in audio_field:
                    path = audio_field['path']
                elif 'array' in audio_field:
                    # Audio already loaded as array
                    raise ValueError("Audio already loaded - expected path")
                else:
                    raise ValueError(f"Unknown audio dict structure: {audio_field.keys()}")
            elif isinstance(audio_field, str):
                path = audio_field
            else:
                raise ValueError(f"Unexpected audio field type: {type(audio_field)}")
            
            dataset_labels = row['labels']  # These are dataset IDs
            
        except Exception as e:
            print(f"\n❌ Error accessing dataset at index {idx}")
            print(f"   Row type: {type(row)}")
            print(f"   Row keys: {row.keys() if hasattr(row, 'keys') else 'N/A'}")
            if 'audio' in row:
                print(f"   Audio type: {type(row['audio'])}")
                print(f"   Audio value: {row['audio']}")
            raise e
        
        # Convert Dataset IDs -> Original Model IDs
        original_model_ids = []
        try:
            for ds_id in dataset_labels:
                ds_id = int(ds_id)
                if ds_id in self.ds_to_original_model:
                    original_model_ids.append(self.ds_to_original_model[ds_id])
        except:
            pass

        # Load and process audio
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

        return {"audio": chunks, "label_id": original_model_ids}

# ==========================================
# 3. INFERENCE FUNCTIONS
# ==========================================

def run_inference_pretrained(model, feature_extractor, dataset, target_indices):
    """
    Inference for PRETRAINED model (full 9700 classes).
    Only evaluates on classes present in target_indices.
    
    Args:
        model: Pretrained model with 9700 classes
        feature_extractor: Feature extractor
        dataset: InferenceDataset (returns original model IDs)
        target_indices: List of original model IDs to evaluate (e.g., [241, 318, ...])
    
    Returns:
        top1_acc, top3_acc, cmAP, AUROC, class_acc (dict)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                       collate_fn=inference_collate_fn)
    
    # Create mapping: Original Model ID -> Evaluation Index (0-131)
    original_to_eval_idx = {oid: i for i, oid in enumerate(target_indices)}
    
    print(f"\n🔍 Inference Setup (Pretrained Model):")
    print(f"  Model has {model.config.num_labels} output classes")
    print(f"  Evaluating on {len(target_indices)} target classes")
    print(f"  Will extract columns: {target_indices[:5]}... from model output")
    
    y_true_eval_indices = []  # Ground truth in eval space (0-131)
    y_probs_eval_space = []   # Predictions in eval space (132 classes)
    
    print(f"\nRunning Inference on {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            raw_chunks = batch['audio'][0]
            original_model_ids = batch['label_id'][0]  # e.g., [241, 318]
            
            if not original_model_ids or len(original_model_ids) == 0:
                continue
            
            # Convert to evaluation indices
            eval_indices = []
            for oid in original_model_ids:
                if oid in original_to_eval_idx:
                    eval_indices.append(original_to_eval_idx[oid])
            
            if not eval_indices:
                continue
            
            # Run model on chunks
            file_probs = []
            for i in range(0, len(raw_chunks), CONFIG["chunk_batch_size"]):
                mb = raw_chunks[i : i + CONFIG["chunk_batch_size"]]
                inputs = feature_extractor(mb, padding=True, return_tensors="pt")
                inputs = inputs.to(CONFIG["device"])
                
                outputs = model(inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu()
                file_probs.append(probs)
            
            if not file_probs:
                continue
            
            # Max voting across chunks
            full_probs, _ = torch.max(torch.cat(file_probs), dim=0)  # Shape: [9700]
            
            # Extract ONLY the columns we care about
            subset_probs = full_probs[target_indices].numpy()  # Shape: [132]
            
            # Renormalize
            subset_probs = subset_probs / (np.sum(subset_probs) + 1e-12)
            
            y_true_eval_indices.append(eval_indices)
            y_probs_eval_space.append(subset_probs)
    
    if len(y_true_eval_indices) == 0:
        print(" No valid samples found!")
        return 0.0, 0.0, 0.0, 0.0, {}
    
    # Compute metrics
    return _compute_metrics(y_true_eval_indices, y_probs_eval_space, 
                           target_indices, original_to_eval_idx)


def run_inference_reduced(model, feature_extractor, dataset, target_indices):
    """
    Inference for REDUCED model (post-surgery, 132 classes).
    Model outputs are already in the correct order (0-131).
    
    Args:
        model: Reduced model with len(target_indices) classes
        feature_extractor: Feature extractor
        dataset: InferenceDataset (returns original model IDs)
        target_indices: List of original model IDs (for mapping and per-class acc)
    
    Returns:
        top1_acc, top3_acc, cmAP, AUROC, class_acc (dict)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                       collate_fn=inference_collate_fn)
    
    # Create mapping: Original Model ID -> New Model Index (0-131)
    original_to_new_idx = {oid: i for i, oid in enumerate(target_indices)}
    
    print(f"\n Inference Setup (Reduced Model):")
    print(f"  Model has {model.config.num_labels} output classes")
    print(f"  Evaluating on {len(target_indices)} target classes")
    print(f"  Model outputs are already in correct order (0-{len(target_indices)-1})")
    
    y_true_new_indices = []  # Ground truth in new space (0-131)
    y_probs_new_space = []   # Predictions in new space (132 classes)
    
    print(f"\nRunning Inference on {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            raw_chunks = batch['audio'][0]
            original_model_ids = batch['label_id'][0]  # e.g., [241, 318]
            
            if not original_model_ids or len(original_model_ids) == 0:
                continue
            
            # Convert to new model indices
            new_indices = []
            for oid in original_model_ids:
                if oid in original_to_new_idx:
                    new_indices.append(original_to_new_idx[oid])
            
            if not new_indices:
                continue
            
            # Run model on chunks
            file_probs = []
            for i in range(0, len(raw_chunks), CONFIG["chunk_batch_size"]):
                mb = raw_chunks[i : i + CONFIG["chunk_batch_size"]]
                inputs = feature_extractor(mb, padding=True, return_tensors="pt")
                inputs = inputs.to(CONFIG["device"])
                
                outputs = model(inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu()
                file_probs.append(probs)
            
            if not file_probs:
                continue
            
            # Max voting across chunks
            final_probs, _ = torch.max(torch.cat(file_probs), dim=0)  # Shape: [132]
            
            y_true_new_indices.append(new_indices)
            y_probs_new_space.append(final_probs.numpy())
    
    if len(y_true_new_indices) == 0:
        print(" No valid samples found!")
        return 0.0, 0.0, 0.0, 0.0, {}
    
    # Compute metrics
    return _compute_metrics(y_true_new_indices, y_probs_new_space,
                           target_indices, original_to_new_idx)


def _compute_metrics(y_true_indices, y_probs, target_indices, original_to_idx_map):
    """
    Shared metric computation for both inference functions.
    
    Args:
        y_true_indices: List of lists of indices (in 0-N space)
        y_probs: List of probability arrays (shape [N])
        target_indices: List of original model IDs
        original_to_idx_map: Dict mapping original_id -> 0-N index
    """
    y_score = np.array(y_probs)
    n_classes = len(target_indices)
    
    # Convert to binary format
    mlb = MultiLabelBinarizer(classes=range(n_classes))
    y_true_binary = mlb.fit_transform(y_true_indices)
    
    print(f"\n Computing metrics on {len(y_true_indices)} samples...")
    print(f"  y_true_binary shape: {y_true_binary.shape}")
    print(f"  y_score shape: {y_score.shape}")
    
    # Top-1 Accuracy
    top1_preds = np.argmax(y_score, axis=1)
    correct_mask = y_true_binary[np.arange(len(y_true_binary)), top1_preds] == 1
    top1_acc = np.mean(correct_mask)
    
    # Top-3 Accuracy
    k = min(3, n_classes)
    topk_preds = np.argsort(y_score, axis=1)[:, -k:][:, ::-1]
    topk_hits = 0
    for i in range(len(y_true_binary)):
        if not set(y_true_indices[i]).isdisjoint(topk_preds[i]):
            topk_hits += 1
    top3_acc = topk_hits / len(y_true_binary)
    
    # cmAP
    try:
        cmAP = average_precision_score(y_true_binary, y_score, average='macro')
    except Exception as e:
        print(f"  cmAP failed: {e}")
        cmAP = 0.0
    
    # AUROC
    try:
        AUROC = roc_auc_score(y_true_binary, y_score, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"  AUROC failed: {e}")
        AUROC = 0.0
    
    # Per-Class Accuracy (return with original IDs as keys)
    class_acc = {}
    for i, original_id in enumerate(target_indices):
        relevant_samples = (y_true_binary[:, i] == 1)
        if np.sum(relevant_samples) > 0:
            matches = (top1_preds[relevant_samples] == i)
            class_acc[original_id] = np.mean(matches)
        else:
            class_acc[original_id] = 0.0
    
    print(f" Top-1: {top1_acc:.4f} | Top-3: {top3_acc:.4f} | cmAP: {cmAP:.4f} | AUROC: {AUROC:.4f}")
    
    return top1_acc, top3_acc, cmAP, AUROC, class_acc

# ==========================================
# 4. SURGERY FUNCTIONS
# ==========================================

def perform_weight_surgery(model, target_indices):
    """Slices the classification head to keep only target classes."""
    print(f"\n{'='*60}")
    print(f"PERFORMING WEIGHT SURGERY")
    print(f"{'='*60}")
    
    with torch.no_grad():
        old_num_protos = model.head.prototype_vectors.shape[0]
        
        k = CONFIG["base_k"]
        indices_to_keep = []
        for idx in target_indices:
            start = idx * k
            indices_to_keep.extend(range(start, start + k))
        
        new_num_protos = len(indices_to_keep)
        new_num_classes = len(target_indices)
        
        print(f"\n Surgery Plan:")
        print(f"  Prototypes: {old_num_protos} -> {new_num_protos}")
        print(f"  Classes: {model.config.num_labels} -> {new_num_classes}")
        
        # 1. Slice prototype_vectors
        print(f"\n Slicing prototype_vectors...")
        old_protos = model.head.prototype_vectors.data
        model.head.prototype_vectors = nn.Parameter(old_protos[indices_to_keep])
        print(f"  ✓ New shape: {model.head.prototype_vectors.shape}")
        
        # 2. Slice ALL buffers
        print(f"\n Slicing prototype-related buffers:")
        buffers_to_update = {}
        for name, buffer in list(model.head.named_buffers()):
            if buffer.shape[0] == old_num_protos:
                print(f"  - {name}: {buffer.shape} -> ", end="")
                buffers_to_update[name] = buffer[indices_to_keep]
                print(f"{buffers_to_update[name].shape}")
        
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
        print(f"\n Updating model metadata...")
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
        
        # 6. Update id2label (for display purposes)
        print(f"\n Creating new label mappings...")
        old_id2label = model.config.id2label
        new_id2label = {}
        new_label2id = {}
        
        for new_id, old_id in enumerate(target_indices):
            # Handle both string and int keys
            if str(old_id) in old_id2label:
                label_name = old_id2label[str(old_id)]
            elif old_id in old_id2label:
                label_name = old_id2label[old_id]
            else:
                label_name = f"class_{old_id}"
            
            new_id2label[new_id] = label_name
            new_label2id[label_name] = new_id
        
        model.config.id2label = new_id2label
        model.config.label2id = new_label2id
        print(f"  ✓ Updated id2label: {len(new_id2label)} classes")
    
    print(f"\n Surgery Complete!")
    print(f"{'='*60}\n")
    return model.to(CONFIG["device"])


def perform_prototype_surgery(model, target_indices, hard_classes, original_id2label):
    """Increases prototypes from 5 to 10 for hard classes."""
    print(f"\n{'='*60}")
    print(f"PERFORMING DYNAMIC PROTOTYPE SURGERY")
    print(f"{'='*60}")
    
    device = model.device
    base_k = CONFIG["base_k"]
    
    print(f"\n Prototype Allocation:")
    class_counts = {}
    
    for i, original_id in enumerate(target_indices):
        is_hard = original_id in hard_classes
        k = CONFIG["hard_k"] if is_hard else base_k
        class_counts[i] = k
        
        # Get name from original mapping
        if str(original_id) in original_id2label:
            name = original_id2label[str(original_id)]
        elif original_id in original_id2label:
            name = original_id2label[original_id]
        else:
            name = f"class_{original_id}"
        
        status = "HARD" if is_hard else " ormal"
        print(f"  {name:<30} : {k:2d} protos ({status})")
    
    # Calculate totals
    total_new = sum(class_counts.values())
    c, h, w = model.head.prototype_vectors.shape[1:]
    old_num_protos = model.head.prototype_vectors.shape[0]
    
    print(f"\n Surgery Plan:")
    print(f"  Total prototypes: {old_num_protos} -> {total_new}")
    print(f"  Hard classes: {len(hard_classes)}")
    
    # Create new tensors
    new_protos = torch.zeros((total_new, c, h, w), device=device)
    new_identity = torch.zeros((total_new, len(target_indices)), device=device)
    
    # Prepare to expand buffers
    buffers_to_expand = {}
    for name, buffer in model.head.named_buffers():
        if buffer.shape[0] == old_num_protos and name != 'prototype_class_identity':
            new_shape = list(buffer.shape)
            new_shape[0] = total_new
            buffers_to_expand[name] = torch.zeros(new_shape, device=device)
    
    # Transplant & Clone
    print(f"\n Cloning prototypes...")
    old_protos = model.head.prototype_vectors.data
    current_idx = 0
    
    for i in range(len(target_indices)):
        target_k = class_counts[i]
        old_start = i * base_k
        old_block = old_protos[old_start : old_start + base_k]
        
        # Copy original
        new_protos[current_idx : current_idx + base_k] = old_block
        
        # Copy buffers
        for name, new_buffer in buffers_to_expand.items():
            old_buffer = dict(model.head.named_buffers())[name]
            old_buffer_block = old_buffer[old_start : old_start + base_k]
            new_buffer[current_idx : current_idx + base_k] = old_buffer_block
        
        # Clone additional for hard classes
        if target_k > base_k:
            needed = target_k - base_k
            for n in range(needed):
                clone = old_block[n % base_k].clone()
                noise = torch.randn_like(clone) * 0.02
                new_protos[current_idx + base_k + n] = clone + noise
                
                for name, new_buffer in buffers_to_expand.items():
                    old_buffer = dict(model.head.named_buffers())[name]
                    old_buffer_block = old_buffer[old_start : old_start + base_k]
                    new_buffer[current_idx + base_k + n] = old_buffer_block[n % base_k].clone()
        
        new_identity[current_idx : current_idx + target_k, i] = 1.0
        current_idx += target_k
    
    # Update Model
    print(f"\n Updating model...")
    model.head.prototype_vectors = nn.Parameter(new_protos)
    
    for name, new_buffer in buffers_to_expand.items():
        delattr(model.head, name)
        model.head.register_buffer(name, new_buffer)
        print(f"  Buffer '{name}': {new_buffer.shape}")
    
    if hasattr(model.head, 'prototype_class_identity'):
        delattr(model.head, 'prototype_class_identity')
    model.head.register_buffer('prototype_class_identity', new_identity)
    print(f"  prototype_class_identity: {new_identity.shape}")
    
    # Rebuild last_layer
    model.head.last_layer = nn.Linear(total_new, len(target_indices), bias=False)
    model.head.last_layer.weight.data.copy_(new_identity.t())
    model.head.last_layer.weight.requires_grad = False
    print(f"  last_layer: [{total_new} x {len(target_indices)}]")
    
    # Update counts
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
    """Fine-tune the model prototypes."""
    print(f"\n{'='*60}")
    print(f"FINE-TUNING MODEL")
    print(f"{'='*60}")
    
    # Freeze backbone, train only prototypes
    trainable_params = []
    for name, param in model.named_parameters():
        if "head" in name and "prototype_vectors" in name:
            param.requires_grad = True
            trainable_params.append(param)
            print(f" Training: {name}")
        else:
            param.requires_grad = False
    
    optimizer = torch.optim.AdamW(trainable_params, lr=CONFIG["lr_head"])
    criterion = nn.CrossEntropyLoss()
    
    # Mapping for training
    original_to_new = {target_indices[i]: i for i in range(len(target_indices))}
    
    def train_collate(batch):
        audio_arrays = []
        valid_labels = []
        
        for item in batch:
            audio_chunks = item["audio"]
            original_ids = item["label_id"]
            
            if not original_ids:
                continue
            
            new_indices = [original_to_new[oid] for oid in original_ids if oid in original_to_new]
            if not new_indices:
                continue
            
            if len(audio_chunks) > 0:
                audio_arrays.append(audio_chunks[0])
                valid_labels.append(new_indices[0])
        
        if len(audio_arrays) == 0:
            return None, None
        
        inputs = feature_extractor(audio_arrays, padding=True, return_tensors="pt")
        labels = torch.tensor(valid_labels, dtype=torch.long)
        
        return inputs, labels
    
    # Create dataloaders
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
            print("  💾 Best model saved!")
    
    # Plot
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
# 5. MAIN PIPELINE
# ==========================================

def main():
    print("\n" + "="*60)
    print("AUDIOPROTONET DYNAMIC PROTOTYPE SURGERY PIPELINE")
    print("="*60 + "\n")
    
    # Load dataset
    print(" Loading dataset...")
    dataset = load_dataset(
        name="PER",
        path='DBD-research-group/BirdSet',
        trust_remote_code=True,
        cache_dir='/home/users/ntu/ytong005/scratch/Birdset_PER'
    )
    dataset = dataset.rename_column("ebird_code_multilabel", "labels")
    dataset = dataset.select_columns(["audio", "labels"])
    
    # Load model
    print(" Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model"],
        trust_remote_code=True
    )
    model.to(CONFIG["device"])
    
    # Build mappings
    print(" Building label mapping...")
    ds_to_original_model, target_indices = build_label_mapping(
        CONFIG["dataset_json"],
        model.config.label2id
    )
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        CONFIG["base_model"],
        trust_remote_code=True
    )
    
    print(f"\n Target classes: {len(target_indices)}")
    
    # Create datasets
    test_ds = InferenceDataset(dataset["test_5s"], ds_to_original_model, train_mode=False)
    print(f" Test samples: {len(test_ds)}")
    
    # Save original id2label before any modifications
    original_id2label = model.config.id2label.copy()
    
    # ==========================================
    # STEP 1: BASELINE (Pretrained Model)
    # ==========================================
    print("\n" + "="*60)
    print("STEP 1: BASELINE INFERENCE (Pretrained Model)")
    print("="*60)
    
    base_top1, base_top3, base_cmAP, base_auroc, base_class_acc = \
        run_inference_pretrained(model, feature_extractor, test_ds, target_indices)
    
    # Identify hard classes
    hard_classes = []
    print(f"\n Per-Class Accuracy (Threshold: {CONFIG['acc_threshold']:.0%}):")
    print(f"{'Class':<30} {'Accuracy':<10} {'Status'}")
    print("-" * 55)
    
    for oid in target_indices:
        acc = base_class_acc[oid]
        
        # Get name from original mapping
        if str(oid) in original_id2label:
            name = original_id2label[str(oid)]
        elif oid in original_id2label:
            name = original_id2label[oid]
        else:
            name = f"class_{oid}"
        
        status = "Good" if acc >= CONFIG["acc_threshold"] else "HARD"
        print(f"{name:<30} {acc:>6.2%}     {status}")
        
        if acc < CONFIG["acc_threshold"]:
            hard_classes.append(oid)
    
    print(f"\n📊 Summary: {len(hard_classes)} hard classes found")
    
    # ==========================================
    # STEP 2: WEIGHT SURGERY
    # ==========================================
    model = perform_weight_surgery(model, target_indices)
    
    # ==========================================
    # STEP 3: POST-SURGERY EVALUATION
    # ==========================================
    print("\n" + "="*60)
    print("STEP 2: POST-SURGERY EVALUATION (Reduced Model)")
    print("="*60)
    
    reduced_top1, reduced_top3, reduced_cmAP, reduced_auroc, reduced_class_acc = \
        run_inference_reduced(model, feature_extractor, test_ds, target_indices)
    
    # ==========================================
    # STEP 4: DYNAMIC PROTOTYPE SURGERY
    # ==========================================
    model = perform_prototype_surgery(model, target_indices, hard_classes, original_id2label)
    
    # ==========================================
    # STEP 5: POST-DYNAMIC SURGERY EVALUATION
    # ==========================================
    print("\n" + "="*60)
    print("STEP 3: POST-DYNAMIC SURGERY EVALUATION")
    print("="*60)
    
    dynamic_top1, dynamic_top3, dynamic_cmAP, dynamic_auroc, dynamic_class_acc = \
        run_inference_reduced(model, feature_extractor, test_ds, target_indices)
    
    # ==========================================
    # STEP 6: FINE-TUNING
    # ==========================================
    model = finetune_model(model, dataset["train"], ds_to_original_model,
                          target_indices, feature_extractor)
    
    # ==========================================
    # STEP 7: FINAL EVALUATION
    # ==========================================
    print("\n" + "="*60)
    print("STEP 4: FINAL EVALUATION (After Fine-tuning)")
    print("="*60)
    
    final_top1, final_top3, final_cmAP, final_auroc, final_class_acc = \
        run_inference_reduced(model, feature_extractor, test_ds, target_indices)
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Metric':<25} {'Baseline':<12} {'Reduced':<12} {'Dynamic':<12} {'Finetuned'}")
    print("-" * 73)
    print(f"{'Top-1 Accuracy':<25} {base_top1:>10.4f}  {reduced_top1:>10.4f}  {dynamic_top1:>10.4f}  {final_top1:>10.4f}")
    print(f"{'Top-3 Accuracy':<25} {base_top3:>10.4f}  {reduced_top3:>10.4f}  {dynamic_top3:>10.4f}  {final_top3:>10.4f}")
    print(f"{'cmAP':<25} {base_cmAP:>10.4f}  {reduced_cmAP:>10.4f}  {dynamic_cmAP:>10.4f}  {final_cmAP:>10.4f}")
    print(f"{'AUROC':<25} {base_auroc:>10.4f}  {reduced_auroc:>10.4f}  {dynamic_auroc:>10.4f}  {final_auroc:>10.4f}")
    print("="*60)
    
    # Save
    torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "dynamic_proto_model_final.pth"))
    print(f"\n Final model saved to {CONFIG['save_dir']}")
    print("\n Pipeline Complete!\n")

if __name__ == "__main__":
    main()