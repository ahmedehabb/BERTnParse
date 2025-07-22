# Bert-n-Parse

## Dependency Graph Extractor
This script preprocesses Arabic sentences from the CAMeL-Lab/BAREC-Shared-Task-2025-sent dataset and extracts linguistic features, including:
- Morphological features
- POS tags
- Universal Dependencies (UD)
- Dependency graphs

It also uses an external tool CamelParser2.0 to parse each sentence into CoNLL format and saves outputs in structured JSON files for downstream usage (e.g., modeling, analysis, or graph-based processing).


### How to Run
Inside the camel_parser directory: 

```
python dependency_graph_extractor.py <Split>
```
Where <Split> must be one of: train - validation - test

```
python dependency_graph_extractor.py train
```

This will:
- Load the train split of the dataset.
- Normalize and clean each sentence.
- Split the data into chunks.
- Call text_to_conll_cli.py on each chunk to generate syntactic parses.
- Extract:

    - Morphological features
    - POS tags
    - UD features
    - Dependency graphs

Save all processed outputs under a folder named splits_<Split>

### Output Files (saved in ./splits_<Split>)
- File Name	Description
- sentences_i_j.txt	Raw cleaned sentence chunks passed to the parsing tool.
- output_i_j.txt	Parsed output of each sentence chunk (in CoNLL format).
- all_parsed_blocks_<Split>.txt	All CoNLL-formatted blocks merged into one file.
- features_list_<Split>.json	Extracted features (e.g., sentence length, UD statistics) per sentence.
- pos_tags_dict_<Split>.json	Mapping of sentence → list of POS tags.
- morph_features_dict_<Split>.json	Mapping of sentence → morphological features (e.g., aspect, gender, number).
- dep_graph_<Split>.json	Dependency graphs for each sentence, in a JSON-serializable structure.

### Notes
Sentences are de-diacritized using pyarabic.araby.strip_diacritics.
Duplicate sentences are dropped before processing.
Each sentence is parsed using the CLI tool before feature extraction.
Feature dictionaries use the cleaned sentence as the key for merging.

## Model Training: BERT + Graph Neural Networks

Once the linguistic features and dependency graphs have been extracted and saved (see [Dependency Graph Extractor](#dependency-graph-extractor)), the next stage involves training a **graph-based readability classifier** on top of BERT embeddings using PyTorch Geometric.

All training logic is implemented in [`training/train.py`](./training/train.py).

---

### 🧠 Model Overview

The model architecture consists of:
- **BERT encoder** (`aubmindlab/bert-base-arabertv02`) to generate contextual token embeddings.
- **Graph Neural Network (GNN)** that operates over dependency graphs, encoding syntactic structure.
- **CORAL-based ordinal classifier** for predicting Arabic readability levels (1–19).

You can choose between two GNN architectures:
- `ImprovedSimpleEdgeGNN` – simple but effective edge-based model.
- `StrongerEdgeGNN` – deeper and more expressive, with multi-head attention and dropout.

---

### 🔄 End-to-End Pipeline

The `train.py` script performs the following steps:

#### 1. **Load & Preprocess the Dataset**
- Loads all splits from the [CAMeL-Lab/BAREC-Shared-Task-2025-sent](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Shared-Task-2025-sent) dataset using Hugging Face Datasets.
- Normalizes Arabic text (removes diacritics using `pyarabic`).
- Ensures proper spacing after punctuation for better dependency parsing.
- Adjusts labels to be 0-indexed (`Readability_Level_19 - 1`).

#### 2. **Attach Dependency Features**
- Each sentence is augmented with:
  - Dependency labels
  - POS tags
  - Head indices
- This data is merged with previous CoNLL-style parses using string-based matching.

#### 3. **Build Graph Representations**
- Dependency graphs are converted into `torch_geometric.data.Data` objects.
- Edge labels use `dep2id`; node features use `pos2id`.
- All graphs are padded so they have equal-length node feature vectors.

#### 4. **Cache Graphs**
- Graphs are built once per split and cached under a `cache/` folder as `.pt` files (e.g., `graph_list_train.pt`).
- Re-running the script reuses cached files unless deleted or modified.

#### 5. **Model Training**
- Trains the GNN over BERT embeddings using CORAL loss for ordinal classification.
- Evaluation is done on the internal test set each epoch.
- Best models (based on accuracy and Quadratic Weighted Kappa) are saved:
  - `model_best_acc.pt`
  - `model_best_qwk.pt`
  - `last_model.pt`

#### 6. **Final Evaluation**
- After training, the best model is evaluated on the **official blind test split**.
- Predictions are saved to a file using `output_to_file`.

---

### 🛠️ How to Run

From the root of the project:

```bash
python training/train.py
```

Make sure you've already:
- Installed required dependencies (see below).
- Parsed all splits using the [`camel_parser`](./camel_parser) pipeline.

---

### 📦 Dependencies

You can install all dependencies using:

```bash
pip install transformers pyarabic regex coral-pytorch camel-tools torch_geometric
pip install -U datasets huggingface_hub
```

---

### 📁 Directory Structure (Relevant Files)

```
.
├── camel_parser/
│   └── .... 
│   └── dependency_graph_extractor.py
│
├── training/
│   ├── train.py  ← Full model training pipeline
│   ├── models.py
│   ├── utils.py
│   ├── graph_utils.py
│   ├── attach_morph_features.py
│   ├── attach_dep_features.py
│   └── dependency_relations.py
│
├── cache/
│   └── graph_list_{split}.pt  ← Cached graph data per dataset split
```

---

### 📝 Notes

- This setup is optimized for experimentation and reuse.
- You can skip morph features if you prefer a cleaner dependency-based model.
- The GNN expects properly padded graphs — this is handled automatically via `pad_pos_tags`.
