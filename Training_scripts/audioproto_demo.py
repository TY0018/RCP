from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
import librosa
import torch
from datasets import Audio
import numpy as np
from tqdm import tqdm

import io
import os
import subprocess
import boto3
from botocore.config import Config
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv, find_dotenv

import json

_ = load_dotenv(find_dotenv())

SAMPLE_RATE = 32000
VARIANT = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-trained model
# ---- MODEL ----
feature_extractor = AutoFeatureExtractor.from_pretrained(
    f"DBD-research-group/AudioProtoPNet-{VARIANT}-BirdSet-XCL", trust_remote_code=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    f"DBD-research-group/AudioProtoPNet-{VARIANT}-BirdSet-XCL", trust_remote_code=True
)
model.eval().to(device)

# ---- Load cutoff time ----
LAST_TIME_FILE = "/home/users/ntu/ytong005/AudioProto/last_processed_time.txt"


def load_cutoff_time(default_time=None):
    """Return last processed timestamp if file exists, else return default."""
    if os.path.exists(LAST_TIME_FILE):
        with open(LAST_TIME_FILE, "r") as f:
            ts_str = f.read().strip()
            if ts_str:
                return datetime.fromisoformat(ts_str)
    return datetime(2025, 8, 21, tzinfo=timezone.utc)


CUTOFF_TIME = load_cutoff_time()

# ---- S3 CONFIGURATION ----
BUCKET_NAME = os.environ["S3_BUCKET"]
BATCH_SIZE = 8
# Connect to S3
endpoint = os.getenv("S3_ENDPOINT")

region = (
    os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or ("auto" if endpoint else "us-east-1")
)

cfg = Config(
    signature_version="s3v4",
    retries={"max_attempts": 5, "mode": "standard"},
    s3={"addressing_style": "virtual"},
)
s3 = boto3.client("s3", endpoint_url=endpoint, region_name=region, config=cfg)


def list_audio_files(bucket=BUCKET_NAME, cutoff_time=CUTOFF_TIME):
    """List all audio files in the S3 bucket modified after cutoff_time."""
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket)

    audio_files = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            if obj["LastModified"] > cutoff_time:
                audio_files.append(
                    {
                        "key": obj["Key"],
                        "last_modified": obj["LastModified"],
                    }  # TODO: fetch station_id
                )
    return audio_files


def load_audio_from_s3(bucket, key):
    """Load an audio file from S3 into memory."""
    file_obj = s3.get_object(Bucket=bucket, Key=key)
    audio_bytes = io.BytesIO(file_obj["Body"].read())
    audio, sr = librosa.load(audio_bytes, sr=SAMPLE_RATE)
    return audio, sr


# ---- DB CONFIGURATION ----
conn = psycopg2.connect(
    host=os.environ["PGHOST"], 
    port=os.environ["PGPORT"],
    dbname=os.environ["PGDATABASE"],
    user=os.environ["PGUSER"],
    password=os.environ["PGPASSWORD"],
    sslmode="require",
)
cursor = conn.cursor()


# ---- INFERENCE ----
def predict_audio_batch(audio_batch):
    """Run model inference on a batch of audio arrays."""
    inputs = feature_extractor(audio_batch, return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
    return probs


# ---- PIPELINE ----
audio_rows = list_audio_files(BUCKET_NAME, CUTOFF_TIME)
print(f"Found {len(audio_rows)} audio files to process before {CUTOFF_TIME}")

# Save latest processed time
latest_time = max(obj["last_modified"] for obj in audio_rows) if audio_rows else datetime.now(timezone.utc)
with open(LAST_TIME_FILE, "w") as f:
    f.write(latest_time.isoformat())

# Get label to name mapping
name_to_label_file = "/home/users/ntu/ytong005/dateset_json/label2name.json"
with open(name_to_label_file, "r") as f:
    parsed = json.load(f)

results = []
k = 3
for i in tqdm(range(0, len(audio_rows), BATCH_SIZE)):
    batch_rows = audio_rows[i : i + BATCH_SIZE]
    audio_batch = []
    valid_rows = []

    for row in batch_rows:
        try:
            audio, sr = load_audio_from_s3(BUCKET_NAME, row["key"])
            audio_batch.append(audio)
            valid_rows.append(
                {"key": row["key"], "last_modified": row["last_modified"]}
            )
        except Exception as e:
            print(f"❌ Failed to load {row['key']}: {e}")

    if not audio_batch:
        continue

    batch_probs = predict_audio_batch(audio_batch)

    # Decode top-k predictions
    id2label = model.config.id2label
    for row, probs in zip(valid_rows, batch_probs):
        topk = np.argsort(probs)[::-1][:k]
        # get created_time
        date_parts = row["key"].split("_")
        if len(date_parts) >= 2:
            date, time = date_parts[0], date_parts[1]
            try:
                created_time = datetime.strptime(date + time, "%Y-%m-%d%H-%M-%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError as e:
                print("❌ Failed to parse date from filename:", date_parts)
                created_time = row["last_modified"]
        else:
            created_time = row["last_modified"]
        result = {
            "recording": row["key"],
            "station_id": "4",  # TODO: fetch from s3
            "processed_time": datetime.now(timezone.utc).isoformat(),
            "created_time": created_time.isoformat(),
            "download_url": f"https://pub-de82980aaf1f4695b8ea6e38d89ed609.r2.dev/{row['key']}",
            **{
                f"label_{i+1}": id2label[topk[i]] for i in range(k)
            },  # TODO: could convert it to the full name of the bird
            **{f"score_{i+1}": float(probs[topk[i]]) for i in range(k)},
            "recording_duration": len(audio_batch[0]) / SAMPLE_RATE,
        }
        for i in range(k):
            id = result[f"label_{i+1}"]
            name = parsed[id]
            en_name = name.split("_")[1]
            result[f"label_{i+1}"] = en_name
        results.append(result)
# response = s3.list_objects_v2(Bucket=BUCKET_NAME)
# audio_path = ...
# label = ...
# audio, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)


# # Visualise waveform
# librosa.display.waveshow(y=audio, sr=sample_rate)
# plt.title(label)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude (normalized)')
# plt.show()

# # Convert to spectrogram
# mel_spec_norm = feature_extractor(audio)
# mel_spec_norm = mel_spec_norm.to(device)
# # Visualisation
# plt.imshow(mel_spec_norm[0,0], origin='lower')
# plt.title('Spectrogram ' + label)
# plt.xlabel('Time (ms)')
# plt.ylabel('Frequency (Mels)')
# plt.show()

# Inference
# with torch.no_grad():
#     output = model(mel_spec_norm)

# Output probabilities
# k = 5

# p = torch.sigmoid(output[0]).cpu()
# top_n_probs, top_n_indices = torch.topk(p, k=k, dim=-1)
# print("top_n_probs: ", top_n_probs)

# label2id = model.config.label2id
# id2label = model.config.id2label

# print(f'Selected species with confidence:')
# print(f"{label:<7} - {probabilities[:, label2id[label]].item():.2%}")
# print(f"Top {k} Predictions with confidence:")
# for idx, conf in zip(top_n_indices.squeeze(), top_n_probs.squeeze()):
#     print(f"{id2label[idx.item()]:<7} - {conf:.2%}")

# ----------------- STORE RESULTS -----------------
CREATE_TABLE_QUERY = """ 
    CREATE TABLE IF NOT EXISTS bird_recordings (
        recording TEXT PRIMARY KEY,
        station_id TEXT NOT NULL,
        processed_time TIMESTAMPTZ NOT NULL,
        created_time TIMESTAMPTZ NOT NULL,
        download_url TEXT NOT NULL,
        label_1 TEXT,
        score_1 DOUBLE PRECISION,
        label_2 TEXT,
        score_2 DOUBLE PRECISION,
        label_3 TEXT,
        score_3 DOUBLE PRECISION,
        recording_duration DOUBLE PRECISION
    );
"""

INSERT_ROWS_QUERY = """
    INSERT INTO bird_recordings (
        recording, station_id, processed_time, created_time, download_url,
        label_1, score_1, label_2, score_2, label_3, score_3, recording_duration
    )
    VALUES %s
    ON CONFLICT (recording) DO UPDATE
    SET station_id = EXCLUDED.station_id,
        processed_time = EXCLUDED.processed_time,
        created_time = EXCLUDED.created_time,
        download_url = EXCLUDED.download_url,
        label_1 = EXCLUDED.label_1,
        score_1 = EXCLUDED.score_1,
        label_2 = EXCLUDED.label_2,
        score_2 = EXCLUDED.score_2,
        label_3 = EXCLUDED.label_3,
        score_3 = EXCLUDED.score_3,
        recording_duration = EXCLUDED.recording_duration;
"""


def batch_insert_results(conn, results):
    data = [
        (
            r["recording"],
            r["station_id"],
            r["processed_time"],
            r["created_time"],
            r["download_url"],
            r["label_1"],
            r["score_1"],
            r["label_2"],
            r["score_2"],
            r["label_3"],
            r["score_3"],
            r["recording_duration"],
        )
        for r in results
    ]
    with conn.cursor() as cur:
        execute_values(cur, INSERT_ROWS_QUERY, data)
    conn.commit()


batch_insert_results(conn, results)

print("✅ Batch processing complete. Processed results:")
print(results[:3])  # show sample