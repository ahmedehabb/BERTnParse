import os
import pandas as pd
import torch
import pyarabic.araby as araby

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch_geometric.loader import DataLoader
from training.attach_morph_features import attach_morphological_features, build_vocab_for_all_features, morph_features, map_morph_str_to_ids
from training.models import BERTGraphModel, ImprovedSimpleEdgeGNN, StrongerEdgeGNN
from training.utils import evaluate, output_to_file
from training.cleaning import clean_example, add_space_after_punctuation_if_not
from training.dependency_relations import build_vocab_for_dependency_relations, build_vocab_for_pos_tags_in_dependency_relations
from training.attach_dep_features import attach_dep_train, attach_dep_eval, attach_dep_test, attach_dep_blind_test
from training.graph_utils import load_or_build_graphs, pad_pos_tags

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_DISABLED"] = "true"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
# Load sentence-level data from the "Dev" split
train_dataset = load_dataset("CAMeL-Lab/BAREC-Shared-Task-2025-sent", split="train")
eval_dataset = load_dataset("CAMeL-Lab/BAREC-Shared-Task-2025-sent", split="validation")
test_dataset = load_dataset("CAMeL-Lab/BAREC-Shared-Task-2025-sent", split="test")

# Load sentence-level data from the "official_blind" split
token = "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"
blind_test_dataset = load_dataset("CAMeL-Lab/BAREC-Shared-Task-2025-BlindTest-sent", token=token, split="test")

# Fix labels to be 0-indexed
train_dataset = train_dataset.map(lambda x: {"labels": x["Readability_Level_19"] - 1, "Sentence": araby.strip_diacritics(x["Sentence"])})
eval_dataset = eval_dataset.map(lambda x: {"labels": x["Readability_Level_19"] - 1, "Sentence": araby.strip_diacritics(x["Sentence"])})
test_dataset = test_dataset.map(lambda x: {"labels": x["Readability_Level_19"] - 1, "Sentence": araby.strip_diacritics(x["Sentence"])})
blind_test_dataset = blind_test_dataset.map(lambda x: {"Sentence": araby.strip_diacritics(x["Sentence"])})

df = pd.DataFrame(train_dataset)
df.head(100)


# Clean sentence first from dataset to be able to merge 
train_dataset = train_dataset.map(clean_example)
eval_dataset = eval_dataset.map(clean_example)
test_dataset = test_dataset.map(clean_example)
blind_test_dataset = blind_test_dataset.map(clean_example)

# --------------------------------------------------------------------------------------------------------------- #

# Attach all morphological features for all sentences saved in files: "morph_features_dict_{split}.csv"
# example: 
# # text = الأربعاء 15 يونيو 2011م
# 1	الأربعاء	أربعاء	NOM	_	ud=NOUN|pos=noun_prop|prc3=0|prc2=0|prc1=0|prc0=Al_det|enc0=0|asp=na|vox=na|mod=na|gen=m|num=s|stt=d|cas=n|per=na|rat=i|token_type=baseword	0	---	_	_
# 2	15	15	NOM	_	ud=NUM|pos=digit|prc3=na|prc2=na|prc1=na|prc0=na|enc0=na|asp=na|vox=na|mod=na|gen=na|num=na|stt=na|cas=na|per=na|rat=na|token_type=baseword	1	MOD	_	_
# 3	يونيو	يونيو	PROP	_	ud=PROPN|pos=noun_prop|prc3=0|prc2=0|prc1=0|prc0=0|enc0=0|asp=na|vox=na|mod=na|gen=m|num=s|stt=i|cas=u|per=na|rat=i|token_type=baseword	2	IDF	_	_
# 4	2011م	2011م	NOM	_	ud=NUM|pos=digit|prc3=na|prc2=na|prc1=na|prc0=na|enc0=na|asp=na|vox=na|mod=na|gen=na|num=na|stt=na|cas=na|per=na|rat=na|token_type=baseword	3	MOD	_	_

# You can choose which features you want to keep, for example: ud, pos, prc3, prc2, prc1, prc0, enc0, asp, vox, mod, gen, num, stt, cas, per, rat, token_type
# I didnt use them in the end, but you can use them if you want by uncommenting the next line

# print("Attach morph features for dataset")
# train_dataset = train_dataset.map(attach_morphological_features)

# print("Attach morph features for eval_dataset")
# eval_dataset = eval_dataset.map(attach_morphological_features)

# print("Attach morph features for test_dataset")
# test_dataset = test_dataset.map(attach_morphological_features)

# print("Attach morph features for blind_test_dataset")
# blind_test_dataset = blind_test_dataset.map(attach_morphological_features)

# # Here we build vocabulary for all morphological features, like mapping from each morphological tag to an id
# feature_vocabs = build_vocab_for_all_features(train_dataset, morph_features)

# # Map morphological features from string to ids
# train_dataset = train_dataset.map(map_morph_str_to_ids, batched=True, fn_kwargs={"feature_vocabs": feature_vocabs, "feature_names": morph_features})
# eval_dataset = eval_dataset.map(map_morph_str_to_ids, batched=True, fn_kwargs={"feature_vocabs": feature_vocabs, "feature_names": morph_features})
# test_dataset = test_dataset.map(map_morph_str_to_ids, batched=True, fn_kwargs={"feature_vocabs": feature_vocabs, "feature_names": morph_features})
# # blind_test_dataset = blind_test_dataset.map(map_morph_str_to_ids, batched=True, fn_kwargs={"feature_vocabs": feature_vocabs, "feature_names": morph_features})

# morph_columns = [f"{feat}_ids" for feat in morph_features]
# # Set torch format for syntactic_feats to be correctly read by collator
# train_dataset.set_format("torch", columns=morph_columns, output_all_columns=True)
# eval_dataset.set_format("torch", columns=morph_columns, output_all_columns=True)
# test_dataset.set_format("torch", columns=morph_columns, output_all_columns=True)
# blind_test_dataset.set_format("torch", columns=morph_columns, output_all_columns=True)


# --------------------------------------------------------------------------------------------------------------- #

#  Attach dependency features to each example in the dataset
# Apply map with appropriate dict
print("Attach dep for train dataset")
train_dataset = train_dataset.map(attach_dep_train)

print("Attach dep for eval dataset")
eval_dataset = eval_dataset.map(attach_dep_eval)

print("Attach dep for test dataset")
test_dataset = test_dataset.map(attach_dep_test)

print("Attach dep for blind test dataset")
blind_test_dataset = blind_test_dataset.map(attach_dep_blind_test)

# --------------------------------------------------------------------------------------------------------------- #

# Adding space after punctuation if not exists in the sentence, because we did that while dependency parsing
# the sentences, in order to make sure the dependency parsing works correctly
# Because in arabic, after comma, semicolon, question mark, etc. there should be a space

# Apply map with appropriate dict
print("add_space_after_punctuation_if_not for train dataset")
train_dataset = train_dataset.map(add_space_after_punctuation_if_not)

print("add_space_after_punctuation_if_not for eval dataset")
eval_dataset = eval_dataset.map(add_space_after_punctuation_if_not)

print("add_space_after_punctuation_if_not for test dataset")
test_dataset = test_dataset.map(add_space_after_punctuation_if_not)

print("add_space_after_punctuation_if_not for blind test dataset")
blind_test_dataset = blind_test_dataset.map(add_space_after_punctuation_if_not)

# --------------------------------------------------------------------------------------------------------------- #
# Building the graphs from examples in the dataset

# First, build dependency relations from training dataset
dep2id = build_vocab_for_dependency_relations(train_dataset)
dep_id2label = {v: k for k, v in dep2id.items()}  # invert your dep2id dict to get labels from ids

# build pos tags relations from training dataset
pos2id = build_vocab_for_pos_tags_in_dependency_relations(train_dataset)
id2pos = {idx: pos for pos, idx in pos2id.items()} # invert your pos2id dict to get labels from ids

# Build/load graphs for each split
graph_list_train = load_or_build_graphs(train_dataset, "train", dep2id, pos2id)
graph_list_eval = load_or_build_graphs(eval_dataset, "eval", dep2id, pos2id)
graph_list_test = load_or_build_graphs(test_dataset, "test", dep2id, pos2id)
graph_list_blind_test = load_or_build_graphs(blind_test_dataset, "blind_test", dep2id, pos2id)

# --------------------------------------------------------------------------------------------------------------- #

# Pad pos_tag_ids to the maximum length in the dataset
# This is necessary because the GNN expects all graphs to have the same number of nodes (i.e., same length of pos_tag_ids)
# We will pad with a special "PAD"
graph_list_train = pad_pos_tags(graph_list_train, pos2id)
graph_list_eval = pad_pos_tags(graph_list_eval, pos2id)
graph_list_test = pad_pos_tags(graph_list_test, pos2id)
graph_list_blind_test = pad_pos_tags(graph_list_blind_test, pos2id)


for data in graph_list_train:
    if hasattr(data, 'edge_attr') and data.edge_attr.dim() == 1:
        data.edge_attr = data.edge_attr.unsqueeze(1)  # add feature dimension

for data in graph_list_eval:
    if hasattr(data, 'edge_attr') and data.edge_attr.dim() == 1:
        data.edge_attr = data.edge_attr.unsqueeze(1)  # add feature dimension

for data in graph_list_test:
    if hasattr(data, 'edge_attr') and data.edge_attr.dim() == 1:
        data.edge_attr = data.edge_attr.unsqueeze(1)  # add feature dimension

for data in graph_list_blind_test:
    if hasattr(data, 'edge_attr') and data.edge_attr.dim() == 1:
        data.edge_attr = data.edge_attr.unsqueeze(1)  # add feature dimension


# --------------------------------------------------------------------------------------------------------------- #
# Now we can use these graph lists with DataLoader for training and evaluation

batch_size = 64
# train_loader = DataLoader(graph_list_train, batch_size=batch_size, shuffle=True)
# val_loader   = DataLoader(graph_list_eval, batch_size=batch_size)
# test_loader  = DataLoader(graph_list_test, batch_size=batch_size)

# To make it faster and experiment better !!
# train_loader = DataLoader(graph_list_train[:int(len(graph_list_train))], batch_size=batch_size, shuffle=True)
# val_loader   = DataLoader(graph_list_eval[:int(len(graph_list_eval))], batch_size=batch_size)
test_loader  = DataLoader(graph_list_test[:int(len(graph_list_test))], batch_size=batch_size)
blind_test_loader  = DataLoader(graph_list_blind_test[:int(len(graph_list_blind_test))], batch_size=batch_size)

# now we will train on train,val
# validate on internal test
# final testing on the official blind test
combined_train_val = graph_list_train + graph_list_eval  # combine lists
combined_train_loader = DataLoader(combined_train_val, batch_size=batch_size, shuffle=True)

# --------------------------------------------------------------------------------------------------------------- #


NUM_CLASSES = 19  # since 1-19
bert_model_ckpt = "aubmindlab/bert-base-arabertv02"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_ckpt)
bert_model = AutoModel.from_pretrained(bert_model_ckpt).to(device)

# Define the GNN model, you can choose between ImprovedSimpleEdgeGNN or StrongerEdgeGNN, both are giving near results
# gnn_model = ImprovedSimpleEdgeGNN(input_dim=768, hidden_dim=512, num_classes=NUM_CLASSES, num_relations=len(dep2id), num_pos_tags=len(pos2id), conv_layers=16).to(device)
gnn_model = StrongerEdgeGNN(input_dim=768, hidden_dim=512, num_classes=NUM_CLASSES, num_relations=len(dep2id), num_pos_tags=len(pos2id), pos_emb_dim=32, dropout=0.1, conv_layers=16, num_heads=4).to(device)

# Create the BERTGraphModel
model = BERTGraphModel(gnn_model, bert_tokenizer, bert_model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Freeze the lower layers of bert (first 8 of 12):
for name, param in bert_model.named_parameters():
    if name.startswith("encoder.layer.") and int(name.split(".")[2]) < 8:
        param.requires_grad = False

# Print the number of trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

# --------------------------------------------------------------------------------------------------------------- #
# Training and Evaluation

num_epochs = 10
best_acc = 0.0
best_qwk = 0.0

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    for data in combined_train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        # For CORAL, loss is returned from model
        loss = out["loss"]

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_acc = evaluate(model, test_loader)
    val_accuracy = val_acc['accuracy']
    val_qwk = val_acc['qwk']

    # Save best accuracy model
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), 'model_best_acc.pt')
        print(f"✅ Saved new best accuracy model: {best_acc:.4f}")

    # Save best QWK model
    if val_qwk > best_qwk:
        best_qwk = val_qwk
        torch.save(model.state_dict(), 'model_best_qwk.pt')
        print(f"✅ Saved new best QWK model: {best_qwk:.4f}")

    print(f"Epoch {epoch:02d}, Loss: {total_loss:.4f}, "
          f"Val Acc: {val_accuracy:.4f}, Val QWK: {val_qwk:.4f}")

torch.save(model.state_dict(), 'last_model.pt')

# --------------------------------------------------------------------------------------------------------------- #
# Load the best model for final evaluation on the blind test set
# model.load_state_dict(torch.load('last_model.pt'))
# model.load_state_dict(torch.load('model_best_qwk.pt'))
# model.load_state_dict(torch.load('model_best_acc.pt'))


output_to_file(model, blind_test_loader)
