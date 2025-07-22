import torch
from torch_geometric.data import Data
import os

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
batch_size = 256
torch.cuda.empty_cache()

def build_graphs_batch_v2(batch, dep2id, pos2id):
    batch_sentences = batch["Sentence_space_after_punct"]
    batch_words = [s.split() for s in batch_sentences]
    batch_size = len(batch_words)

    graph_list = []
    for i in range(batch_size):
        words = batch_words[i]
        sentence = batch["Sentence"][i]
        dep_graph = batch["dep_graph"][i]
        labels = batch["labels"][i] if "labels" in batch else None
        sample_id = batch["ID"][i]

        edge_index = [[], []]
        edge_attr = []

        if dep_graph:
            for edge in dep_graph["edges"]:
                src = edge["source"]
                tgt = edge["target"]
                dep = edge.get("dep", "---")
                if src <= len(words) and tgt <= len(words):
                    edge_index[0].append(src)
                    edge_index[1].append(tgt)
                    edge_attr.append(dep2id.get(dep, 0))

        # Extract POS tag ids per node (including ROOT node)
        pos_tag_ids_per_node = []
        pos_tag_ids_per_node.append([pos2id.get("ROOT", -1)])  # Root POS

        for node in dep_graph["nodes"]:
            pos_tags = node.get("pos_tags", [])
            ids = [pos2id[pos] for pos in pos_tags if pos in pos2id]
            pos_tag_ids_per_node.append(ids if ids else [])

        max_len = max(len(ids) for ids in pos_tag_ids_per_node)
        padded_pos_tag_ids = [
            ids + [pos2id.get("PAD", -1)] * (max_len - len(ids)) for ids in pos_tag_ids_per_node
        ]
        
        graph = Data(
            # Do NOT include x here; model will compute it
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.long),
            pos_tag_ids=torch.tensor(padded_pos_tag_ids, dtype=torch.long),
            sentence=sentence,
            sentence_tokenized=words,  # pass to model for BERT
            # y=torch.tensor(labels, dtype=torch.long),
            id=torch.tensor(sample_id, dtype=torch.long),
            num_nodes = len(words) + 1,  # +1 for the ROOT node
        )

        # to make sure it works for blind testset too
        if labels is not None:
            graph.y = torch.tensor(labels, dtype=torch.long)
        else:
            # place holder for official blindtestset, 0 : because it must belong to 0, numclasses - 1
            graph.y = torch.tensor([0], dtype=torch.long)


        graph_list.append(graph)

    return graph_list

def load_or_build_graphs(dataset, split_name, dep2id, pos2id):
    cache_path = os.path.join(CACHE_DIR, f"graph_list_{split_name}.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached graphs for {split_name} from {cache_path}")
        return torch.load(cache_path, weights_only=False)
    
    print(f"Building graphs for {split_name} dataset")
    graph_list = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        graph_list.extend(build_graphs_batch_v2(batch, dep2id, pos2id))
    torch.save(graph_list, cache_path)
    print(f"Saved graphs for {split_name} to {cache_path}")
    return graph_list

# Function to pad POS tag IDs in the graph list
# This ensures that all graphs have the same number of POS tag IDs per node
def pad_pos_tags(graph_list, pos2id):
    max_len = max(data.pos_tag_ids.size(1) for data in graph_list)
    pad_id = pos2id.get("PAD", -1)
    for data in graph_list:
        cur_len = data.pos_tag_ids.size(1)
        if cur_len < max_len:
            padding = torch.full(
                (data.pos_tag_ids.size(0), max_len - cur_len),
                pad_id,
                dtype=data.pos_tag_ids.dtype,
                device=data.pos_tag_ids.device
            )
            data.pos_tag_ids = torch.cat([data.pos_tag_ids, padding], dim=1)
    return graph_list