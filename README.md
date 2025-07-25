# Bert-n-Parse

**Bert-n-Parse** is a system for sentence-level **Arabic readability assessment**, developed as part of our submission to **Task 1 of the BAREC Shared Task 2025**. The system integrates **contextual** and **syntactic** information by combining:

- Pretrained BERT embeddings (`aubmindlab/bert-base-arabertv02`)
- Graph Neural Networks (GNNs) operating over **dependency parse trees**

Our key hypothesis is that readability depends not just on **lexical choice**, but also on **syntactic complexity**—particularly in morphologically rich languages like Arabic. To capture this, each sentence is represented as a **dependency graph** where:

- **Nodes** carry BERT-based contextual embeddings, POS tags embeddings, .. (morphological features)
- **Edges** reflect grammatical dependencies

The GNN models the structure of these graphs, enabling the system to learn both **deep linguistic features** and **contextual patterns**. Empirically, our syntax-aware model improves over a strong BERT-only baseline, demonstrating the importance of structural information in fine-grained readability prediction.


## 🧠 Dependency Graph Extractor

This script preprocesses Arabic sentences from the [CAMeL-Lab/BAREC-Shared-Task-2025-sent](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Shared-Task-2025-sent) dataset and extracts rich linguistic features for graph-based modeling. It leverages the [CamelParser v2.0](https://github.com/CAMeL-Lab/camel_parser) for syntactic analysis and outputs structured data for downstream machine learning or analysis tasks.

### 🔍 Extracted Features
- **Morphological features**
- **Part-of-speech (POS) tags**
- **Universal Dependencies (UD)**
- **Dependency graph structure (head/deprel info)**

---

### 🧩 Token Alignment and Word-Level Graph Merging

A crucial step in the preprocessing pipeline is harmonizing the granularity mismatch between AraBERTv2's **subword tokenization** and the **word-level outputs** from the CamelParser. Arabic words often contain rich morphology (e.g., clitics or affixes like `سأصيدهما` → `س+أصيد+هما`), and the dependency parser returns linguistically meaningful segments that do not align directly with BERT’s subword units.

To ensure consistent node representations between the parser and BERT embeddings:
- **Subword embeddings** from AraBERTv2 are **averaged** to produce a single vector per word.
- On the graph side, **clitic and morpheme-level nodes** are **merged** into unified word-level nodes.
- Edges between merged nodes are also consolidated to preserve syntactic structure.

This alignment ensures a **1-to-1 mapping** between BERT vectors and graph nodes, enabling meaningful message passing in GNNs.

📌 See **Figure 1** for an illustration of this merging process:

![Merging Example](figures/merging.png)

---

### 🚀 How to Run

Inside the `camel_parser/` directory:

```bash
python dependency_graph_extractor.py <Split>
```

Replace `<Split>` with one of:

- `train`
- `validation`
- `test`

Example:

```bash
python dependency_graph_extractor.py train
```

This will:
1. Load the dataset split (e.g., `train`).
2. Normalize and clean the Arabic sentences.
3. Split the data into manageable chunks.
4. Use `text_to_conll_cli.py` to parse each chunk using CamelParser.
5. Extract and save:
   - Morphological and syntactic features
   - Dependency relations
   - Sentence-level statistics

💡 Alternatively, you can run the full pipeline using the provided notebook:  
📓 **`Camel_parser.ipynb`** – includes all steps from preprocessing to dependency graph extraction in one place.

---

### 📦 Output Files (Saved in `./splits_<Split>/`)

| File Name                       | Description                                                  |
|--------------------------------|--------------------------------------------------------------|
| `sentences_i_j.txt`            | Cleaned input sentences for parser chunk `i_j`               |
| `output_i_j.txt`               | Parser output (CoNLL format) for chunk `i_j`                 |
| `all_parsed_blocks_<Split>.txt`| All CoNLL-formatted parses merged into one file             |
| `features_list_<Split>.json`   | Extracted statistics and metadata per sentence               |
| `pos_tags_dict_<Split>.json`   | Mapping from sentence → list of POS tags                     |
| `morph_features_dict_<Split>.json` | Mapping from sentence → morphological features           |
| `dep_graph_<Split>.json`       | Dependency graphs as JSON-serializable structures            |

---

### 📝 Notes

- All sentences are **de-diacritized** using `pyarabic.araby.strip_diacritics`.
- Duplicate sentences are **removed before processing**.
- Parsing is done via **CamelParser2.0** using its CLI interface.
- Each output file is keyed by the **cleaned sentence string** to ensure consistent merging.

### 🔗 Dependency Parsing Tool
Dependency parsing in this project was performed using **CamelParser2.0: A State-of-the-Art Dependency Parser for Arabic** (Elshabrawy et al., 2023).

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

📌 **Figure 2** below summarizes the full system architecture, from dependency parsing and token alignment to graph-based classification:

![System Architecture](figures/system_arch.png)

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

💡 Alternatively, you can run the full end-to-end pipeline (from data loading to final evaluation) using the provided notebook:  
📓 **`barec-model.ipynb`** – mirrors all steps in `train.py` in one place for easy experimentation and interactive debugging.

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
│   ├── another_models.py  : These are some other models based on bert only used for experiments.
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

---

## 📊 Results

We evaluate our model using **Quadratic Weighted Kappa (QWK)**, the primary metric for ordinal classification. Our syntax-aware approach achieves higher QWK scores on both internal and official test sets compared to a strong AraBERTv2 baseline, demonstrating the effectiveness of integrating syntactic structure.

![Results Comparison](figures/results.png)

---
### 📚 Citation

If you find the Bert-n-Parse useful in your research, please cite: 

> Your Name  
> *Title of Your Work*  
> Conference/Journal, Year  
> DOI or URL (if available)
