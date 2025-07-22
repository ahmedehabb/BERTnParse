import json
from functools import partial


with open("/kaggle/input/outputs-dep/dep_graph_train.json", "r") as f:
    dep_dict_train = json.load(f)

with open("/kaggle/input/outputs-dep/dep_graph_validation.json", "r") as f:
    dep_dict_eval = json.load(f)

with open("/kaggle/input/outputs-dep/dep_graph_test.json", "r") as f:
    dep_dict_test = json.load(f)

with open("/kaggle/input/outputs-dep/dep_graph_blind_testset.json", "r") as f:
    dep_dict_blind_test = json.load(f)

def attach_dep_features(example, dep_dict):
    sentence = example["Sentence"]
    dep_graph = dep_dict.get(sentence, None)
    if dep_graph == None:
        print("cant find dep graph for sentence:", example["Sentence"])
    example["dep_graph"] = dep_graph
    return example

# Create partials with the correct feature dicts
attach_dep_train = partial(attach_dep_features, dep_dict=dep_dict_train)
attach_dep_eval = partial(attach_dep_features, dep_dict=dep_dict_eval)
attach_dep_test = partial(attach_dep_features, dep_dict=dep_dict_test)
attach_dep_blind_test = partial(attach_dep_features, dep_dict=dep_dict_blind_test)

