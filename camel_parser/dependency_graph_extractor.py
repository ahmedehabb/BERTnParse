import re
import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # or XGBoost if you want
from sklearn.metrics import classification_report
from datasets import load_dataset
from datasets import Dataset
import pyarabic.araby as araby
from utils import get_upos, extract_features_from_parse, extract_pos_tags_from_block, extract_features_from_ud, extract_morph_features_from_block, get_each_output_from_file, extract_sentence_from_block, build_dep_graph
from clean_sentence_utils import clean_broken_arabic_words, clean_sentence, clean_example


# Load sentence-level data from the "Dev" split
train_dataset = load_dataset("CAMeL-Lab/BAREC-Shared-Task-2025-sent", split="train")
eval_dataset = load_dataset("CAMeL-Lab/BAREC-Shared-Task-2025-sent", split="validation")
test_dataset = load_dataset("CAMeL-Lab/BAREC-Shared-Task-2025-sent", split="test")

# Fix labels to be 0-indexed
train_dataset = train_dataset.map(lambda x: {"labels": x["Readability_Level_19"] - 1, "Sentence": araby.strip_diacritics(x["Sentence"])})
eval_dataset = eval_dataset.map(lambda x: {"labels": x["Readability_Level_19"] - 1, "Sentence": araby.strip_diacritics(x["Sentence"])})
test_dataset = test_dataset.map(lambda x: {"labels": x["Readability_Level_19"] - 1, "Sentence": araby.strip_diacritics(x["Sentence"])})

# we should always clean each sentence in the dataset to be able to join with features
train_dataset = train_dataset.map(clean_example)
eval_dataset = eval_dataset.map(clean_example)
test_dataset = test_dataset.map(clean_example)

df = pd.DataFrame(train_dataset)

if Split == "test":
    dataset = test_dataset
elif Split == "validation":
    dataset = eval_dataset
else:
    dataset = train_dataset


# Convert to pandas first
df = dataset.to_pandas()

# Drop duplicates by sentence
df_unique = df.drop_duplicates(subset=["Sentence"])

# Convert back to Hugging Face Dataset if needed
dataset = Dataset.from_pandas(df_unique)

N_SPLITS = 10  # Number of chunks
OUTPUT_DIR = "./splits_{}".format(Split)  # Directory to save splits
os.makedirs(OUTPUT_DIR, exist_ok=True)

chunk_size = len(dataset) // N_SPLITS
subsets = []

for i in range(N_SPLITS):
    start = i * chunk_size
    # Make last chunk include all remaining elements
    end = (i + 1) * chunk_size if i < N_SPLITS - 1 else len(dataset)
    subsets.append(dataset.select(range(start, end)))

# Step 2: Process each chunk
N_SPLITS_PER_CHUNK = 1  # Number of sub-chunks per problematic chunk

for i, cleaned_subset in enumerate(subsets):
    subset_len = len(cleaned_subset)
    chunk_size = math.ceil(subset_len / N_SPLITS_PER_CHUNK)

    for j in range(N_SPLITS_PER_CHUNK):
        start_idx = j * chunk_size
        end_idx = min((j + 1) * chunk_size, subset_len)
        sub_chunk = cleaned_subset.select(range(start_idx, end_idx))

        # Save sentences
        input_path = os.path.join(OUTPUT_DIR, f"sentences_{i}_{j+1}.txt")
        with open(input_path, "w", encoding="utf-8") as f:
            for sent in sub_chunk["Sentence"]:
                f.write(sent.strip() + "\n")

        # Run CLI tool
        output_path = os.path.join(OUTPUT_DIR, f"output_{i}_{j+1}.txt")
        os.system(f"python text_to_conll_cli.py -f text -i {input_path} > {output_path}")


# get all files that start with "output_" and end with ".txt"
all_parsed_blocks = []
output_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("output_") and f.endswith(".txt")]
for output_file in output_files:
    print(f"Processing {output_file}...")
    filepath = os.path.join(OUTPUT_DIR, output_file)
    blocks = get_each_output_from_file(filepath)
    all_parsed_blocks.extend(blocks)
    
# Now `all_parsed_blocks` contains everything from all output files
print(f"Total parsed blocks: {len(all_parsed_blocks)}")

# write all_parsed_blocks into a file all_parsed_blocks.txt
with open(f"{OUTPUT_DIR}/all_parsed_blocks_{Split}.txt", "w", encoding="utf-8") as f:
    for block in all_parsed_blocks:
        for line in block:
            f.write(line + "\n")
        f.write("\n")  # Separate blocks with an empty line

features_list = []
for block in all_parsed_blocks:
    sentence = extract_sentence_from_block(block)
    if sentence is not None:
        feats = extract_features_from_parse(block)
        feats["Sentence"] = sentence  # Clean the sentence to match while joining
        features_list.append(feats)

# write features_list to a file features_list.json, its a list of dicts
import json
with open(f"{OUTPUT_DIR}/features_list_{Split}.json", "w", encoding="utf-8") as f:
    json.dump(features_list, f, ensure_ascii=False, indent=4)


# extract all pos tags with sentence from all_parsed_blocks and save sentence as key, pos tags as value
pos_tags_dict = {}
for block in all_parsed_blocks:
    sentence = extract_sentence_from_block(block)
    sentence = sentence
    if sentence is not None:
        pos_tags = extract_pos_tags_from_block(block)
        pos_tags_dict[sentence] = pos_tags

# save file to pos_tags_dict.json
with open(f"{OUTPUT_DIR}/pos_tags_dict_{Split}.json", "w", encoding="utf-8") as f:
    import json
    json.dump(pos_tags_dict, f, ensure_ascii=False, indent=4)


# Features you want to extract
all_features = [
    "ud", "prc3", "prc2", "prc1", "prc0", "enc0", "gen", "num", "cas", "per", "asp", "vox", "mod", "stt", "rat", "token_type"
]

# Main loop over blocks
morph_features_dict = {}
for block in all_parsed_blocks:
    sentence = extract_sentence_from_block(block)
    if sentence is not None:
        morph_features = extract_morph_features_from_block(block, all_features)
        morph_features_dict[sentence] = morph_features

# Save to JSON
output_path = os.path.join(OUTPUT_DIR, f"morph_features_dict_{Split}.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(morph_features_dict, f, ensure_ascii=False, indent=4)

print(f"Saved morphological features to {output_path}")

# Build dependency graphs for all blocks
dep_graphs_dict = {}
for i, block in enumerate(all_parsed_blocks):
    print(f"Processing block {i}...")
    dep_graph = build_dep_graph(block)
    sentence = extract_sentence_from_block(block)
    dep_graphs_dict[sentence] = dep_graph

# Save to JSON
output_path = os.path.join(OUTPUT_DIR, f"dep_graph_{Split}.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dep_graphs_dict, f, ensure_ascii=False, indent=4)

print(f"Saved dep_graphs_dict to {output_path}")