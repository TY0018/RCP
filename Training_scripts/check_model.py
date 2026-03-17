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
# print(model.config)
feature_extractor = AutoFeatureExtractor.from_pretrained(f"DBD-research-group/AudioProtoPNet-{variant}-BirdSet-XCL", trust_remote_code=True)
print(feature_extractor)