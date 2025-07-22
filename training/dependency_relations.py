
# This is an example of the dependency graph, attached to each example in the dataset
# "الأربعاء 15 يونيو 2011م": {
#         "nodes": [
#             {
#                 "id": 1,
#                 "word": "الأربعاء",
#                 "token_ids": [
#                     1
#                 ],
#                 "pos_tags": [
#                     "NOUN"
#                 ]
#             },
#             {
#                 "id": 2,
#                 "word": "15",
#                 "token_ids": [
#                     2
#                 ],
#                 "pos_tags": [
#                     "NUM"
#                 ]
#             },
#             {
#                 "id": 3,
#                 "word": "يونيو",
#                 "token_ids": [
#                     3
#                 ],
#                 "pos_tags": [
#                     "PROPN"
#                 ]
#             },
#             {
#                 "id": 4,
#                 "word": "2011م",
#                 "token_ids": [
#                     4
#                 ],
#                 "pos_tags": [
#                     "NUM"
#                 ]
#             }
#         ],
#         "edges": [
#             {
#                 "source": 0,
#                 "target": 1,
#                 "dep": "---"
#             },
#             {
#                 "source": 1,
#                 "target": 2,
#                 "dep": "MOD"
#             },
#             {
#                 "source": 2,
#                 "target": 3,
#                 "dep": "IDF"
#             },
#             {
#                 "source": 3,
#                 "target": 4,
#                 "dep": "MOD"
#             }
#         ]
#     },

# Here we build a vocabulary for dependency relations, "dep" in the example above
def build_vocab_for_dependency_relations(dataset):
    dep_set = set()
    for example in dataset:
        dep_graph = example.get('dep_graph')
        if dep_graph is None:
            continue
        for edge in dep_graph.get('edges', []):
            dep = edge.get('dep')
            if dep:
                dep_set.add(dep)
    return {dep: idx for idx, dep in enumerate(sorted(dep_set))}

# Function to build a vocabulary for POS tags in dependency relations
# This function will iterate over each example in the dataset and collect all unique POS tags
# It will return a dictionary where keys are POS tags and values are their corresponding indices
# This is useful for mapping POS tags to ids later
def build_vocab_for_pos_tags_in_dependency_relations(dataset):
    pos_set = set()
    for example in dataset:
        dep_graph = example.get('dep_graph')
        if dep_graph is None:
            continue
        for node in dep_graph.get('nodes', []):
            pos_tags = node.get('pos_tags', [])
            for pos in pos_tags:
                if pos:
                    pos_set.add(pos)

    sorted_tags = sorted(pos_set)
    vocab = {'PAD': 0}  # Make sure PAD is first
    for idx, tag in enumerate(sorted_tags, start=1):
        vocab[tag] = idx

    vocab['ROOT'] = len(vocab)  # Add ROOT at the end
    return vocab

