from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
import librosa
import torch
import matplotlib.pyplot as plt
from IPython.display import Audio
import datasets
from datasets import Audio
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm
import json

variant = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(f"DBD-research-group/AudioProtoPNet-{variant}-BirdSet-XCL", trust_remote_code=True)
# Load checkpoint
checkpoint = torch.load("/home/users/ntu/ytong005/model_hard_mined.pth")
model.load_state_dict(checkpoint, strict=True)
model.eval().to(device)

# Load dataset
dataset = datasets.load_dataset(
    name="PER",
    path='DBD-research-group/BirdSet',
    trust_remote_code=True,
    cache_dir='/home/users/ntu/ytong005/scratch/Birdset_PER'
)

sample_rate = 32_000
dataset["test_5s"] = dataset["test_5s"].cast_column(
    column="audio",
    feature=Audio(
        sampling_rate=sample_rate,
        mono=True,
        decode=True,
    ),
)

dataset = dataset.rename_column("ebird_code_multilabel", "labels")
columns_to_keep = {"audio", "filepath", "labels", "detected_events", "start_time", "end_time", "ebird_code", "length"}
removable_test_columns = [
    column for column in dataset["test_5s"].column_names if column not in columns_to_keep
]
dataset["test_5s"] = dataset["test_5s"].remove_columns(removable_test_columns)
print("Dataset :", dataset)
print(dataset["test_5s"][0])

# Load the dataset classes
with open("dataset_json/per_ebird.json", "r") as f:
    dataset_json = json.load(f)
    dataset_id2label = dataset_json["id2label"]

model_label2id = model.config.label2id
valid_class_indices = [
    model_label2id[label] for label in dataset_id2label.values() if label in model_label2id
]

print("Valid class indices for this dataset:", valid_class_indices)

# Run Inference
all_probs = []
all_targets = [] # true classes

feature_extractor = AutoFeatureExtractor.from_pretrained(f"DBD-research-group/AudioProtoPNet-{variant}-BirdSet-XCL", trust_remote_code=True)
for sample in tqdm(dataset["test_5s"], desc="Evaluating"):
    audio = sample["audio"]["array"]
    true_ids = sample["labels"]
    with torch.no_grad():
        mel_spec_norm = feature_extractor(audio)
        mel_spec_norm = mel_spec_norm.to(device)
        output = model(mel_spec_norm)
        logits = output.logits
        # mask logits
        mask = torch.full_like(logits, float('-inf'))  # mask irrelevant classes
        mask[:, valid_class_indices] = logits[:, valid_class_indices]
        
    # Probabilities
    p = torch.sigmoid(mask).cpu()
    probabilities = p.numpy().squeeze()
    all_probs.append(probabilities)
    # Get the top 3 predictions by confidence
    # top_n_probs, top_n_indices = torch.topk(probabilities, k=3, dim=-1)
    target = np.zeros(len(model.config.id2label))
    # True values
    for id in true_ids:
        label = dataset_id2label[str(id)]
        if label in model_label2id:
            model_id = model_label2id[label]
            target[model_id] = 1
    all_targets.append(target)

all_probs = np.array(all_probs)
all_targets = np.array(all_targets)
# print("probabilities: ", all_probs)
# print("targets: ", all_targets)
# print("label to id:", model.config.label2id)


# Calculate metric
print("Calculate metrics")
# cmAP = mean of per-class AP
cmAP = np.mean([
    average_precision_score(all_targets[:, i], all_probs[:, i])
    for i in valid_class_indices
])

# AUROC = mean per-class ROC AUC
try:
    AUROC = np.mean([
        roc_auc_score(all_targets[:, i], all_probs[:, i])
        for i in valid_class_indices
        if len(np.unique(all_targets[:, i])) > 1
    ])
except ValueError:
    AUROC = float("nan")

# T1-acc = Top-1 accuracy (only if single-label)
pred_indices = np.argmax(all_probs[:, valid_class_indices], axis=1)
pred_labels = [valid_class_indices[idx] for idx in pred_indices]
correct = []

for i in range(len(all_targets)):
    true_labels = np.where(all_targets[i] == 1)[0]  # indices of correct classes
    if len(true_labels) == 0:
        continue
    if pred_labels[i] in true_labels:
        correct.append(1)
    else:
        correct.append(0)

t1_acc = np.mean(correct)


# Print final results
print(f"cmAP:  {cmAP:.4f}")
print(f"AUROC: {AUROC:.4f}")
print("T1-Acc:", t1_acc)