# Bert-n-Parse

## Dependency Graph Extractor
This script preprocesses Arabic sentences from the CAMeL-Lab/BAREC-Shared-Task-2025-sent dataset and extracts linguistic features, including:
- Morphological features
- POS tags
- Universal Dependencies (UD)
- Dependency graphs

It also uses an external tool CamelParser2.0 to parse each sentence into CoNLL format and saves outputs in structured JSON files for downstream usage (e.g., modeling, analysis, or graph-based processing).


### How to Run
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