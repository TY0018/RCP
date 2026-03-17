import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import soundfile as sf
import datasets
from datasets import Audio, DatasetDict
import torchaudio
import librosa
from event_mapping import XCEventMapping
# Create output directory


UHH = datasets.load_dataset(
    name="UHH",
    path='DBD-research-group/BirdSet',
    trust_remote_code=True,
    cache_dir='/home/users/ntu/ytong005/scratch/Birdset_UHH'
)
# print("POW:", POW["train"][0])

SNE = datasets.load_dataset(
    name="SNE",
    path='DBD-research-group/BirdSet',
    trust_remote_code=True,
    cache_dir='/home/users/ntu/ytong005/scratch/Birdset_SNE'
)

# NES = datasets.load_dataset(
#     name="NES",
#     path='DBD-research-group/BirdSet',
#     trust_remote_code=True,
#     cache_dir='/home/users/ntu/ytong005/scratch/Birdset_NES'
# )
# print("PER:", PER["train"][0])


#print(type(POW["train"]))
# example = POW['train'][0]
# path = example["audio"]["path"]   # path string
# example_validation = POW['valid'][0]
# print("test: ", POW['test'][0])

# waveform_16k, sr = librosa.load(path, sr=16000, mono=True)
# XCM = XCM.cast_column("audio", Audio(sampling_rate=16000))

# Does not load waveform  into memory during mapping later
# sample_rate = 32_000
# POW["train"] = POW["train"].cast_column(
#     column="audio",
#     feature=Audio(
#         sampling_rate=sample_rate,
#         mono=True,
#         decode=False,
#     ),
# )

# mapper = XCEventMapping()
# POW["train"] = POW["train"].map(
#     mapper,
#     remove_columns=["audio"],
#     batched=True,
#     batch_size=300,
#     num_proc=3,
#     desc="Train event mapping",
# )

# POW["train"] = POW["train"].remove_columns("audio")
# XCM = XCM.rename_column("ebird_code_multilabel", "labels")

# Valid/test dataset processing
# POW["test_5s"] = POW["test_5s"].cast_column(
#     column="audio",
#     feature=Audio(
#         sampling_rate=sample_rate,
#         mono=True,
#         decode=True,
#     ),
# )

# POW["test"] = POW["test"].remove_columns("audio")
# POW = POW.rename_column("ebird_code_multilabel", "labels")

# columns_to_keep = {"audio", "filepath", "labels", "detected_events", "start_time", "end_time", "ebird_code", "length"}

# removable_train_columns = [
#     column for column in POW["train"].column_names if column not in columns_to_keep
# ]
# removable_test_columns = [
#     column for column in POW["test_5s"].column_names if column not in columns_to_keep
# ]

# print(removable_train_columns)
# print(removable_test_columns)
# POW["train"] = POW["train"].remove_columns(removable_train_columns)
# POW["test_5s"] = POW["test_5s"].remove_columns(removable_test_columns)

# print(POW)
# print(POW["train"][0]["filepath"])
# print(POW["train"][0]["detected_events"])
# print(POW["test_5s"][0])
# audio_feature = POW["test_5s"].features["audio"]
# path = POW["test"][0]["audio"]["path"]
# decoded_audio = audio_feature.decode_example({"path": path})
# audio = POW["test_5s"][0]["audio"]["array"]
# librosa.display.waveshow(y=audio, sr=sample_rate)
# plt.title(POW["test_5s"][0]["ebird_code"])
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude (normalized)')
# plt.show()

# Split valid dataset into dev and test
# valid_dataset = XCM["valid"]
# split = valid_dataset.train_test_split(test_size=0.3, seed=42)
# dev_dataset = split["train"]
# test_dataset = split["test"]
# XCM = DatasetDict({
#     "train": XCM["train"],
#     "dev": dev_dataset,
#     "test": test_dataset,
# })

# OUTPUT_DIR = "birdset_preprocessed"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
# os.makedirs(TRAIN_DIR, exist_ok=True)
# TRAIN_RTTM_FILE = os.path.join(TRAIN_DIR, "train.rttm") # num_rows: 649241
# TRAIN_LST_FILE = os.path.join(TRAIN_DIR, "train.lst")
# TRAIN_UEM_FILE = os.path.join(TRAIN_DIR, "train.uem")

# TRAIN_PROC_DIR = os.path.join(OUTPUT_DIR, "train_proc")
# os.makedirs(TRAIN_PROC_DIR, exist_ok=True)
# TRAIN_PROC_RTTM_FILE = os.path.join(TRAIN_PROC_DIR, "train_proc.rttm") # TRAIN PROC ROWS: 431282
# TRAIN_PROC_LST_FILE = os.path.join(TRAIN_PROC_DIR, "train_proc.lst")
# TRAIN_PROC_UEM_FILE = os.path.join(TRAIN_PROC_DIR, "train_proc.uem")

# TRAIN_SMALL_DIR = os.path.join(OUTPUT_DIR, "train_small")
# os.makedirs(TRAIN_SMALL_DIR, exist_ok=True)
# TRAIN_SMALL_RTTM_FILE = os.path.join(TRAIN_SMALL_DIR, "train_small.rttm") # TRAIN SMALL ROWS: 85788
# TRAIN_SMALL_LST_FILE = os.path.join(TRAIN_SMALL_DIR, "train_small.lst")
# TRAIN_SMALL_UEM_FILE = os.path.join(TRAIN_SMALL_DIR, "train_small.uem")

# DEV_DIR = os.path.join(OUTPUT_DIR, "dev")
# os.makedirs(DEV_DIR, exist_ok=True)
# DEV_RTTM_FILE = os.path.join(DEV_DIR, "dev.rttm") # num_rows: 3192
# DEV_LST_FILE = os.path.join(DEV_DIR, "dev.lst")
# DEV_UEM_FILE = os.path.join(DEV_DIR, "dev.uem")

# TEST_DIR = os.path.join(OUTPUT_DIR, "test")
# os.makedirs(TEST_DIR, exist_ok=True)
# TEST_RTTM_FILE = os.path.join(TEST_DIR, "test.rttm") # num_rows: 1368
# TEST_LST_FILE = os.path.join(TEST_DIR, "test.lst")
# TEST_UEM_FILE = os.path.join(TEST_DIR, "test.uem")


# def write_rttm(file_id, segments, out_path):
#     with open(out_path, "a") as f:
#         for seg in segments:
#             start, end, species = seg
#             dur = end - start
#             line = f"SPEAKER {file_id} 1 {start:.2f} {dur:.2f} <NA> <NA> {species} <NA> <NA>\n"
#             f.write(line)

# def write_uem(file_id, segments, out_path):
#     """
#     Write UEM file covering only the annotated event clusters (segments).
#     segments = [(start, end, label), ...]
#     """
#     with open(out_path, "a") as f:
#         for seg in segments:
#             start, end, _ = seg
#             f.write(f"{file_id} NA {start:.2f} {end:.2f}\n")

# train_proc_audio_files = []
# train_audio_files = []
# dev_audio_files = []
# test_audio_files = []
# train_small_audio_files = []

# train_species_time = [0 for _ in range(409)]
# ALL TRAIN
# for i, row in enumerate(XCM["train"]):
#     audio_path = row["filepath"]
#     id = os.path.splitext(os.path.basename(audio_path))[0]
#     file_id = os.path.splitext(os.path.relpath(audio_path, start="/home/users/ntu/ytong005/scratch/Birdset_xcm/downloads/extracted"))[0]
#     if id =="XC387892":
#         print("file", audio_path, "event: ", row["detected_events"], "length: ", row["length"])
#         break
    # if "ebird_code" in row and "detected_events" in row and row["detected_events"]:
    #     segments = []
    #     start, end = row["detected_events"]
    #     if end > row["length"]:
    #         continue
    #     if end - start < 1.0:
    #         continue
    #     species = row["ebird_code"]
    #     if train_species_time[species] >= 600:
    #         continue
    #     # if id == "XC727959":
    #     #     start, end = 0.0, 3.0
    #     train_species_time[species] += (end-start)
    #     segments.append((start, end, species))

    # Write RTTM
    # write_rttm(file_id, segments, TRAIN_SMALL_RTTM_FILE)

    #write uem
    # write_uem(file_id, segments, TRAIN_SMALL_UEM_FILE)

    # Append file_id + path together
    # train_small_audio_files.append(file_id)

# TRAIN_PROC (1.0s)
# for i, row in enumerate(XCM["train"]):
#     audio_path = row["filepath"]
#     # file_id = os.path.splitext(os.path.basename(audio_path))[0]
#     file_id = os.path.splitext(os.path.relpath(audio_path, start="/home/users/ntu/ytong005/scratch/Birdset_xcm/downloads/extracted"))[0]

#     if "ebird_code" in row and "detected_events" in row and row["detected_events"]:
#         segments = []
#         start, end = row["detected_events"]
#         species = row["ebird_code"]
#         segments.append((start, end, species))

#     # Write RTTM
#     write_rttm(file_id, segments, TRAIN_PROC_RTTM_FILE)

#     #write uem
#     write_uem(file_id, segments, TRAIN_PROC_UEM_FILE)

#     # Append file_id + path together
#     train_proc_audio_files.append(file_id)

# with open(TRAIN_SMALL_RTTM_FILE, "r") as f:
#     num_lines = sum(1 for _ in f)
#     print(f"TRAIN SMALL ROWS: {num_lines}")

# # DEV FILES
# for i, row in enumerate(XCM["dev"]):
#     audio_path = row["filepath"]
#     # file_id = os.path.splitext(os.path.basename(audio_path))[0]
#     file_id = os.path.splitext(os.path.relpath(audio_path, start="/home/users/ntu/ytong005/scratch/Birdset_xcm/downloads/extracted"))[0]

#     if "labels" in row and row["labels"]:
#         segments = []
#         start, end = row["start_time"], row["end_time"]
#         for species in row["labels"]:
#             segments.append((start, end, species))

#         # Write RTTM
#         write_rttm(file_id, segments, DEV_RTTM_FILE)

#         #write uem
#         write_uem(file_id, segments, DEV_UEM_FILE)

#     dev_audio_files.append(file_id)

# TEST FILES
# for i, row in enumerate(XCM["test"]):
#     audio_path = row["filepath"]
#     # file_id = os.path.splitext(os.path.basename(audio_path))[0]
#     file_id = os.path.splitext(os.path.relpath(audio_path, start="/home/users/ntu/ytong005/scratch/Birdset_xcm/downloads/extracted"))[0]

#     if "labels" in row and row["labels"]:
#         segments = []
#         start, end = row["start_time"], row["end_time"]
#         for species in row["labels"]:
#             segments.append((start, end, species))

#         # Write RTTM
#         write_rttm(file_id, segments, TEST_RTTM_FILE)

#         #write uem
#         write_uem(file_id, segments, TEST_UEM_FILE)
    
#     test_audio_files.append(file_id)

# create lst files
# def write_lst(audio_files, output_file):
#     with open(output_file, "w") as f:
#         for path in audio_files:
#             f.write(f"{path}\n")

# write_lst(train_audio_files, TRAIN_LST_FILE)
# write_lst(dev_audio_files, DEV_LST_FILE)
# write_lst(test_audio_files, TEST_LST_FILE)
# write_lst(train_proc_audio_files, TRAIN_PROC_LST_FILE)
# write_lst(train_small_audio_files, TRAIN_SMALL_LST_FILE)
