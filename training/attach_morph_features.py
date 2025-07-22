import json

# FEATURE EXTRACTION 

# Define the morph features resulted from dep graph you want (check morph_features_dict_test outputs)
# morph_features = ["prc3", "prc2", "prc1", "prc0", "enc0", "gen", "num", "cas", "per", "asp", "vox", "mod", "stt", "rat", "token_type"]
morph_features = ["gen", "num", "cas", "per", "asp", "vox", "mod", "stt", "rat", "token_type"]

with open("/kaggle/input/outputs-dep/morph_features_dict_train.json", "r") as f:
    morph_dict_train = json.load(f)

with open("/kaggle/input/outputs-dep/morph_features_dict_validation.json", "r") as f:
    morph_dict_eval = json.load(f)

with open("/kaggle/input/outputs-dep/morph_features_dict_test.json", "r") as f:
    morph_dict_test = json.load(f)

# merge dicts
morph_dict = {**morph_dict_train, **morph_dict_eval, **morph_dict_test}

# Function to attach morphological features to each example in the dataset
# This function will iterate over each example in the dataset and attach the morphological features
def attach_morphological_features(example):
    sentence = example["Sentence"]
    
    sentence_features = morph_dict.get(sentence)
    for feature in morph_features:
        if sentence_features:
            feature_values = sentence_features[feature]
            feature_string = " ".join(feature_values)
            example[feature] = feature_string
        else:
            if sentence != "":
                print("cant find morph features for sentence:", sentence)
            example[feature] = ""
    return example


# Build vocabulary for each morph feature, like mapping from each morphological tag to an id
def build_vocab_for_sequence_feature(dataset, feature_name, add_pad_token=True):
    unique_tokens = set()
    for example in dataset:
        tokens = example.get(feature_name)
        if tokens:
            unique_tokens.update(tokens.split())
    sorted_tokens = sorted(unique_tokens)
    
    vocab = {}
    start_idx = 0
    if add_pad_token:
        vocab['PAD'] = start_idx
        start_idx += 1
    
    for idx, token in enumerate(sorted_tokens, start=start_idx):
        vocab[token] = idx
    
    return vocab

# Build vocab for all morph features function that iterates over all morph features and call build_vocab_for_sequence_feature
# This will create a dictionary where each key is a feature name and the value is the corresponding vocabulary
# This is useful to map morph features to ids later
def build_vocab_for_all_features(dataset, feature_names):
    feature_vocabs = {}
    for feat in feature_names:
        vocab = build_vocab_for_sequence_feature(dataset, feat)
        feature_vocabs[feat] = vocab
    return feature_vocabs



# morph_str_to_ids: convert for example if the ud string was: "NOUN NUM PROPN NUM" -> [0, 1, 2, 1] according to the vocab
def morph_str_to_ids(feature_vocabs, feature_name, morph_string):
    return [feature_vocabs[feature_name][tag] for tag in morph_string.split()]

# 99th percentile length for morphemes of a sentence is: 65, so will use max_len = 64
def map_morph_str_to_ids(batch, feature_vocabs, feature_names=morph_features, max_len=64):
    for feat in feature_names:
        all_ids = []
        pad_token = "PAD"
        for morph_str in batch[feat]:
            tags = morph_str.split()
            # pad with "PAD" strings
            if len(tags) > max_len:
                tags = tags[:max_len]
            else:
                tags += [pad_token] * (max_len - len(tags))
            ids = morph_str_to_ids(feature_vocabs, feat, " ".join(tags))
            all_ids.append(ids)
        batch[f"{feat}_ids"] = all_ids
    return batch
