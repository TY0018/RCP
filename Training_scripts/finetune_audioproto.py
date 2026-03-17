import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoFeatureExtractor
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
import matplotlib.pyplot as plt
import json
import librosa
import numpy as np

# ==========================================
# 1. SETUP FOCAL LOSS
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * CE_loss
        return torch.mean(F_loss) if self.reduction == 'mean' else torch.sum(F_loss)

# ==========================================
# 2. LOAD YOUR CHECKPOINT
# ==========================================
variant = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = f"DBD-research-group/AudioProtoPNet-{variant}-BirdSet-XCL"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    trust_remote_code=True
)
trainable_params = []
backbone_params = []
for name, param in model.named_parameters():
    # ...EXCEPT the classifier and prototypes
    if "classifier" in name or "prototype" in name or "head" in name or "projector" in name:
        param.requires_grad = True
        trainable_params.append(param)
        print(f"Training: {name}")
    else:
        param.requires_grad = False
        backbone_params.append(param)

# checkpoint = torch.load("/home/users/ntu/ytong005/model_hard_mined.pth")
# model.load_state_dict(checkpoint, strict=True)
model.to(device)
# model.train()

feature_extractor = AutoFeatureExtractor.from_pretrained(f"DBD-research-group/AudioProtoPNet-{variant}-BirdSet-XCL", trust_remote_code=True)

# ==========================================
# 3. SETUP OPTIMIZER
# ==========================================
lr_prototypes = 1e-4
lr_backbone = 1e-5
optimizer = torch.optim.AdamW(trainable_params, lr=lr_prototypes) # Standard is 1e-4, we use 1e-5

# Use Focal Loss instead of standard CrossEntropy
criterion = FocalLoss(gamma=2.0)

# ==========================================
# 4. TRAIN
# ==========================================
dataset = load_dataset(
    name="PER",
    path='DBD-research-group/BirdSet',
    trust_remote_code=True,
    cache_dir='/home/users/ntu/ytong005/scratch/Birdset_PER'
)
sample_rate = 32_000
# dataset["train"] = dataset["train"].cast_column(
#     column="audio",
#     feature=Audio(
#         sampling_rate=sample_rate,
#         mono=True,
#         decode=True,
#     ),
# )

# dataset = dataset.rename_column("ebird_code_multilabel", "labels")
# columns_to_keep = {"audio", "filepath", "labels", "detected_events", "start_time", "end_time", "ebird_code", "length"}
# removable_train_columns = [
#     column for column in dataset["train"].column_names if column not in columns_to_keep
# ]
# dataset["train"] = dataset["train"].remove_columns(removable_train_columns)
# print("Dataset :", dataset)
# print(dataset["train"][0])

# Filter / Rename Columns
dataset = dataset.rename_column("ebird_code_multilabel", "labels")
# (Simplified column keeping for clarity)
columns_to_keep = ["audio", "labels"]
dataset = dataset.select_columns(columns_to_keep)

# Load the dataset classes
with open("dataset_json/per_ebird.json", "r") as f:
    dataset_json = json.load(f)
    dataset_id2label = dataset_json["id2label"]

model_label2id = model.config.label2id
valid_class_indices = [
    model_label2id[label] for label in dataset_id2label.values() if label in model_label2id
]

print("Valid class indices for this dataset:", valid_class_indices)

# !!!!! Dataset to model id mapping
model_label2id = model.config.label2id 

# 2. Get Dataset's Label Map (Dataset ID -> eBird Code)
with open("dataset_json/per_ebird.json", "r") as f:
    dataset_json = json.load(f)
    # Ensure keys are integers
    dataset_id2label = {int(k): v for k, v in dataset_json["id2label"].items()}

# 3. Build the Translation Dictionary
# Map: Dataset_ID -> eBird_Code -> Model_ID
dataset_id_to_model_id = {}
ignored_indices = []

print(f"Mapping {len(dataset_id2label)} dataset classes to model...")

for ds_id, ds_label in dataset_id2label.items():
    if ds_label in model_label2id:
        model_id = model_label2id[ds_label]
        dataset_id_to_model_id[ds_id] = model_id
    else:
        # This dataset class does not exist in the model!
        # We must ignore it or the training will crash.
        print(f"⚠️ Warning: Dataset class '{ds_label}' (ID {ds_id}) not found in Model. Ignoring.")
        ignored_indices.append(ds_id)

# --- SPLIT TRAIN INTO TRAIN & VAL ---
print("Splitting training set into Train (80%) and Validation (20%)...")
split_data = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_data["train"]
val_dataset = split_data["test"]

# sample = split_data["test"].train_test_split(test_size=0.1, seed=42)
# train_dataset = sample["train"]
# val_dataset = sample["test"] 

# --- C. Create Data Loaders (The Fix for Batching) ---
def collate_fn(batch):
    audio_arrays = []
    valid_labels = []
    # 1. Extract audio arrays
    for x in batch:
        # audio = x["audio"]["array"]
        try:
            file_path = x["audio"]["path"]
            if not file_path:
                print(f"Missing file_path, {x}")
                continue
            audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)


        # 3. Extract Labels
            ds_label = x["labels"][0] if isinstance(x["labels"], list) else x["labels"]
            if ds_label in dataset_id_to_model_id:
                model_id = dataset_id_to_model_id[ds_label]
                audio_arrays.append(audio)
                valid_labels.append(model_id)
            else:
                continue
        except Exception as e:
            print(f"Error loading {x}: {e}")
            continue

    if len(audio_arrays) == 0:
        return None, None
    
    # 3. Featurize (Convert Audio -> Spectrogram -> Tensor)
    inputs = feature_extractor(audio_arrays, padding=True, return_tensors="pt")
    # print(f"Inputs: {inputs}")
    
    return inputs, torch.tensor(valid_labels).long()

# Create standard PyTorch DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn) # Using test as val

# ==================================================================
print("Starting Hard Negative Mining (Focal Loss)...")
# Store metrics for plotting
train_losses = []
val_losses = []
epochs = 15
best_val_loss = float('inf')

TOTAL_EPOCHS = 8

for epoch in range(TOTAL_EPOCHS): # Run for ~5-10 epochs
# --- TRAIN PHASE ---
    model.train()
    running_train_loss = 0.0
    
    if epoch == 5:
        print("Unfreezing backbone")
        for param in model.parameters():
            param.requires_grad = True 
        optimizer = torch.optim.AdamW([
                {'params': trainable_params, 'lr': lr_prototypes}, # Keep Head Fast
                {'params': backbone_params, 'lr': lr_backbone}     # Make Body Slow
            ])
    # Use tqdm for progress bar
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    for inputs, labels in progress:
        if inputs is None: continue
        # inputs = {"pixel_values": inputs.to(device)}
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        # if use_mixup and np.random.random() < 0.5: # Apply Mixup 50% of the time
        #     lam = np.random.beta(1.0, 1.0) # Mixing ratio (e.g., 0.7)
        #     index = torch.randperm(inputs.size(0)).to(device) # Random shuffle of batch

        #     mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
            
        #     # Calculate loss for both sets of labels
        #     outputs = model(mixed_inputs)
            
        #     # Loss = lam * Loss(Label_A) + (1-lam) * Loss(Label_B)
        #     loss = lam * criterion(outputs.logits, labels) + \
        #         (1 - lam) * criterion(outputs.logits, labels[index])
        # else:
        #     # Standard Training
        #     outputs = model(inputs)
        #     loss = criterion(outputs.logits, labels)
        

        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
        progress.set_postfix(loss=loss.item())
        
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # --- VALIDATE ---
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            running_val_loss += loss.item()
            
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Save Model
        torch.save(model.state_dict(), "model_hard_mined.pth")
        print("best model saved")

# ==========================================
# 4. PLOT CURVES
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training vs Validation Loss (Hard Negative Mining)')
plt.xlabel('Epochs')
plt.ylabel('Focal Loss')
plt.legend()
plt.grid(True)
plt.savefig("training_curve.png")
print("Curve saved to training_curve.png")
