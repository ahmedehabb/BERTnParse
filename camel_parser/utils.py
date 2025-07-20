
def get_upos(feat_str):
    # feat_str example: "ud=PROPN|pos=noun_prop|prc3=0|prc2=0|..."
    for feat in feat_str.split('|'):
        if feat.startswith('ud='):
            return feat[len('ud='):]  # Extract "PROPN"
    return None

def extract_features_from_parse(parse_lines):
    """
    Input:
      parse_lines: List[str], lines of the CoNLL-like parse (tokens only, no comment lines)
    Output:
      features: dict with syntactic features
    """
    tokens = []
    heads = []
    deprels = []
    upos_tags = []

    # Parse lines to extract relevant info
    for line in parse_lines:
        if line.startswith('#'):
            continue  # skip comment lines

        parts = line.strip().split('\t')
        if len(parts) < 8:
            continue  # skip malformed lines
        token_id = int(parts[0])
        word = parts[1]
        upos = get_upos(parts[5])
        head = int(parts[6])
        deprel = parts[7]

        tokens.append(word)
        upos_tags.append(upos)
        heads.append(head)
        deprels.append(deprel)

    n = len(tokens)

    # Dependency distances
    dep_distances = [abs((i+1) - head) if head != 0 else 0 for i, head in enumerate(heads)]
    avg_dep_distance = sum(dep_distances) / n if n > 0 else 0
    max_dep_distance = max(dep_distances) if dep_distances else 0
    long_deps_count = sum(d > 3 for d in dep_distances)

    # POS tag counts and ratios
    from collections import Counter
    pos_counts = Counter(upos_tags)

    noun_count = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
    verb_count = pos_counts.get('VERB', 0)
    adj_count = pos_counts.get('ADJ', 0)
    adv_count = pos_counts.get('ADV', 0)
    adp_count = pos_counts.get('ADP', 0)
    pron_count = pos_counts.get('PRON', 0)
    cconj_count = pos_counts.get('CCONJ', 0)
    num_count = pos_counts.get('NUM', 0)
    punct_count = pos_counts.get('PUNCT', 0)
    sconj_count = pos_counts.get('SCONJ', 0)
    part_count = pos_counts.get('PART', 0)
    det_count = pos_counts.get('DET', 0)
    aux_count = pos_counts.get('AUX', 0)
    intj_count = pos_counts.get('INTJ', 0)
    pipe_count = pos_counts.get('|', 0)  # assuming '|' shows up in the pos column

    total_content_words = noun_count + verb_count + adj_count + adv_count
    total_words = n

    verb_to_noun_ratio = (verb_count / noun_count) if noun_count > 0 else 0
    content_word_ratio = (total_content_words / total_words) if total_words > 0 else 0

    # Dependency relation counts
    dep_counts = Counter(deprels)
    sbj_count = dep_counts.get('SBJ', 0)
    obj_count = dep_counts.get('OBJ', 0)
    advcl_count = dep_counts.get('ADVCL', 0)
    acl_count = dep_counts.get('ACL', 0)
    conj_count = dep_counts.get('CONJ', 0)
    cc_count = dep_counts.get('CC', 0)

    # Tree structure
    children = {i+1: [] for i in range(n)}
    root = None
    for i, h in enumerate(heads, 1):
        if h == 0:
            root = i
        else:
            children[h].append(i)

    def tree_depth(node):
        if not children[node]:
            return 1
        else:
            return 1 + max(tree_depth(c) for c in children[node])

    max_depth = tree_depth(root) if root else 0

    dependents_counts = [len(children[h]) for h in children]
    avg_dependents = sum(dependents_counts) / n if n > 0 else 0
    max_dependents = max(dependents_counts) if dependents_counts else 0

    # Left and right dependents
    left_deps = sum(1 for i, h in enumerate(heads, 1) if h != 0 and h > i)
    right_deps = sum(1 for i, h in enumerate(heads, 1) if h != 0 and h < i)
    left_right_dep_ratio = (left_deps / right_deps) if right_deps != 0 else 0

    # Compose features
    features = {
        'avg_dep_distance': avg_dep_distance,
        'max_dep_distance': max_dep_distance,
        'long_deps_count': long_deps_count,
        'noun_count': noun_count,
        'verb_count': verb_count,
        'adj_count': adj_count,
        'adv_count': adv_count,
        'adp_count': adp_count,
        'pron_count': pron_count,
        'cconj_count': cconj_count,
        'num_count': num_count,
        'punct_count': punct_count,
        'sconj_count': sconj_count,
        'part_count': part_count,
        'det_count': det_count,
        'aux_count': aux_count,
        'intj_count': intj_count,
        'pipe_count': pipe_count,
        'verb_to_noun_ratio': verb_to_noun_ratio,
        'content_word_ratio': content_word_ratio,
        'sbj_count': sbj_count,
        'obj_count': obj_count,
        'advcl_count': advcl_count,
        'acl_count': acl_count,
        'conj_count': conj_count,
        'cc_count': cc_count,
        'max_depth': max_depth,
        'avg_dependents': avg_dependents,
        'max_dependents': max_dependents,
        'left_deps': left_deps,
        'right_deps': right_deps,
        'left_right_dep_ratio': left_right_dep_ratio,
        'total_words': total_words
    }

    return features


def extract_pos_tags_from_block(block):
    """
    Input:
      block: List[str], lines of the CoNLL-like parse (tokens only, no comment lines)
    Output:
      pos_tags: List[str] with POS tags
    """
    pos_tags = []
    for line in block:
        if line.startswith('#'):
            continue  # skip comment lines

        parts = line.strip().split('\t')
        parts = line.strip().split('\t')
        if len(parts) < 8:
            continue  # skip malformed lines
        upos = get_upos(parts[5])
        if upos:
            pos_tags.append(upos)
        else:
            pos_tags.append('UNKNOWN')

    return pos_tags

def extract_features_from_ud(ud_string):
    """Parse the 'ud=' string and return a dict of features."""
    if not ud_string.startswith('ud='):
        return {}
    feature_string = ud_string
    features = feature_string.split('|')
    feature_dict = {}
    for feat in features:
        if '=' in feat:
            k, v = feat.split('=', 1)
            feature_dict[k] = v
    return feature_dict

def extract_morph_features_from_block(block, all_features):
    """
    Extract morphological features from one CoNLL block.
    Returns: dict of {feature_name: list of values}
    """
    features_by_name = {feat: [] for feat in all_features}
    
    for line in block:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 8:
            continue
        morph_dict = extract_features_from_ud(parts[5])
        for feat in all_features:
            features_by_name[feat].append(morph_dict.get(feat, "na"))  # "na" for missing
    return features_by_name

def get_each_output_from_file(filepath):
    blocks = []
    current_block = []
    in_block = False

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Start of a new block
            if line.startswith("# text = "):
                if current_block:  # Save previous block if one is in progress
                    blocks.append(current_block)
                current_block = [line]
                in_block = True

            # Skip treeTokens lines
            elif line.startswith("# treeTokens"):
                continue

            # Inside a block, add lines until empty line
            elif in_block:
                if line == "":
                    blocks.append(current_block)
                    current_block = []
                    in_block = False
                else:
                    current_block.append(line)

        # Add last block if file doesn’t end with an empty line
        if current_block:
            blocks.append(current_block)

    return blocks

def extract_sentence_from_block(block):
    if block and block[0].startswith("# text ="):
        return block[0].replace("# text = ", "").strip()
    return None


# TODO:: very important when joining in the inference -> 
# replace the arabic punctuation marks with english ? 
# add space after the arabic punctuation marks if not already there
def build_dep_graph(block):
    """
    Build a dependency graph from a CoNLL block.
    Returns: dict with nodes and edges
    """
    def get_token_type(feat_str):
        if feat_str == "_" or feat_str.strip() == "":
            return {}
        all_feats = dict(item.split("=") for item in feat_str.split("|") if "=" in item)
        return all_feats.get('token_type', 'baseword')  # Default to 'baseword' if not found
    
    sentence = extract_sentence_from_block(block)
    
    if sentence is None:
        return {'nodes': [], 'edges': []}
    
    lines = block[1:]  # Skip the first line which is the sentence line

    tokens_data = []
    for line in lines:
        parts = line.split('\t')
        token_id = int(parts[0])
        form = parts[1]
        if "NOAN" in form:
            # replace "NOAN" with the next part, also replace only "NOAN" not the whole, since for example we have "الNOAN" زنا -> should be "الزنا"
            form = form.replace("NOAN", parts[2])
        head = parts[6]
        dep = parts[7]
        pos_tag = get_upos(parts[5])
        token_type = get_token_type(parts[5])  # Extract token type from the features
        
        tokens_data.append({
            'id': token_id,
            'form': form,
            'head': head,
            'dep': dep,
            'token_type': token_type,
            'pos_tag': pos_tag,  # Add POS tag to the token data
        })

    merged_results = []
    current_word_tokens = []
    current_word_form = ""
    current_word_tokens_contain_baseword = False  # Track if the current word contains a baseword
    map_english_punctuation = {
        ";": ["؛"],  # Arabic semicolon
        ",": ["،", "٫"],  # Arabic comma
        "?": ["؟"],  # Arabic question mark
        "%": ["٪"],  # Arabic percentage sign
        "*": ["۝"],  # Arabic symbol for verse end
    }
    # map_english_punctuation_values = [punct for puncts in map_english_punctuation.values() for punct in puncts]
    # count all the punctuations found in sentence
    punctuations_occurences = []
    for i, letter in enumerate(sentence):
        if letter in map_english_punctuation.keys():
            punctuations_occurences.append(i)

    # Add space after the arabic punctuation marks if its directly attached to the next word
    offset = 0
    for index in punctuations_occurences:
        adjusted_index = index + offset
        if adjusted_index < len(sentence) - 1 and sentence[adjusted_index + 1] != ' ':
            sentence = sentence[:adjusted_index + 1] + ' ' + sentence[adjusted_index + 1:]
            offset += 1  # Adjust offset due to inserted space


    # print(f"Processed sentence: {sentence}")

    # count all the punctuations found in sentence again after adding spaces
    punctuations_occurences = []
    for i, letter in enumerate(sentence):
        if letter in map_english_punctuation.keys():
            punctuations_occurences.append(i)

    print(f"Total punctuations found in sentence: {[sentence[pun] for pun in punctuations_occurences]}")

    current_punctuations_index = 0

    for token in tokens_data:
        form = token['form']
        token_type = token['token_type']
        
        if token_type.startswith('prc'):
            # means a new word is getting started, we need to save the previous word if exists
            if current_word_tokens and current_word_tokens_contain_baseword:
                # Only save if the current word is a baseword
                merged_results.append({
                    'word': current_word_form,
                    'token_ids': [t['id'] for t in current_word_tokens],
                    'heads': [t['head'] for t in current_word_tokens],
                    'deps': [t['dep'] for t in current_word_tokens],
                    'pos_tags': [t['pos_tag'] for t in current_word_tokens],
                    'merged_form': "".join([t['form'].lstrip('+').rstrip('+') for t in current_word_tokens]),
                })
                # Reset current word group
                current_word_tokens = []
                current_word_form = ""

            # Start new word group
            current_word_tokens_contain_baseword = False
            current_word_tokens.append(token)
            current_word_form += form.lstrip('+').rstrip('+')
            
            continue

        if token_type == 'baseword':
            # special case for ";" baseword, sometimes its combined with previous word
            
            if form in map_english_punctuation.keys():
                # we need first to check if its really combined with previous word, need to add all of its properties too
                
                # get the occurence of the punctuation in the sentence
                if current_punctuations_index < len(punctuations_occurences):
                    punctuation_index = punctuations_occurences[current_punctuations_index]
                else:
                    raise Exception(f"Current punctuations index {current_punctuations_index} is out of bounds for sentence: {sentence}")

                # Check if the punctuation is concatenated directly to the previous/next word, or there is a space before it
                # case where the punctuation is directly attached to the next word
                if (punctuation_index < len(sentence) - 1 and sentence[punctuation_index + 1] != ' '):
                    raise Exception(f"Unexpected punctuation '{form}' at index {punctuation_index} in sentence: {sentence}")
                
                if (punctuation_index > 0 and sentence[punctuation_index - 1] != ' '):
                    # print(f"Found '{current_word_form + map_english_punctuation[form]}' in sentence, merging...")
                    # If the current word is part of the sentence, we merge it
                    current_word_tokens.append(token)
                    current_word_form += form.lstrip('+').rstrip('+')

                    # Save previous word if exists
                    if current_word_tokens:
                        merged_results.append({
                            'word': current_word_form,
                            'token_ids': [t['id'] for t in current_word_tokens],
                            'heads': [t['head'] for t in current_word_tokens],
                            'deps': [t['dep'] for t in current_word_tokens],
                            'pos_tags': [t['pos_tag'] for t in current_word_tokens],
                            'merged_form': "".join([t['form'].lstrip('+').rstrip('+') for t in current_word_tokens]),
                        })
                    # Start new word group
                    current_word_tokens = []
                    current_word_form = ""
                    current_word_tokens_contain_baseword = False
                    # increment current_punctuations_index
                    current_punctuations_index += 1
                    continue
                else:
                    # increment current_punctuations_index
                    current_punctuations_index += 1

            # Save previous word if exists
            if current_word_tokens and current_word_tokens_contain_baseword:
                # Only save if the current word is a baseword
                merged_results.append({
                    'word': current_word_form,
                    'token_ids': [t['id'] for t in current_word_tokens],
                    'heads': [t['head'] for t in current_word_tokens],
                    'deps': [t['dep'] for t in current_word_tokens],
                    'pos_tags': [t['pos_tag'] for t in current_word_tokens],
                    'merged_form': "".join([t['form'].lstrip('+').rstrip('+') for t in current_word_tokens]),
                })
                # Reset current word group
                current_word_tokens = []
                current_word_form = ""

            # Start new word group
            current_word_tokens_contain_baseword = True
            current_word_tokens.append(token)
            current_word_form += form.lstrip('+').rstrip('+')
        else:
            # Add clitic or enc0 tokens to current word group
            current_word_tokens.append(token)
            current_word_form += form.lstrip('+').rstrip('+')

    # Append last word group after loop
    if current_word_tokens:
        merged_results.append({
            'word': current_word_form,
            'token_ids': [t['id'] for t in current_word_tokens],
            'heads': [t['head'] for t in current_word_tokens],
            'deps': [t['dep'] for t in current_word_tokens],
            'pos_tags': [t['pos_tag'] for t in current_word_tokens],
            'merged_form': "".join([t['form'].lstrip('+') for t in current_word_tokens]),
        })

    # Map from token_id to merged word index (1-based)
    tokenid_to_merged_idx = {}
    for merged_idx, merged_token in enumerate(merged_results, 1):
        for tid in merged_token['token_ids']:
            tokenid_to_merged_idx[tid] = merged_idx

    # Build heads mapping to merged node indices
    for entry in merged_results:
        heads_graph = []
        for head in entry['heads']:
            head_id = int(head)
            if head_id == 0:
                heads_graph.append(0)  # Root
            else:
                merged_head_idx = tokenid_to_merged_idx.get(head_id, -1)
                heads_graph.append(merged_head_idx)
        entry['Heads_graph'] = heads_graph

    # Construct nodes and edges for graph
    nodes = []
    edges = []

    for idx, entry in enumerate(merged_results, 1):
        nodes.append({
            'id': idx,
            'word': entry['word'],
            'token_ids': entry['token_ids'],
            'pos_tags': entry['pos_tags'],
        })

        for head_idx, dep in zip(entry['Heads_graph'], entry['deps']):
            if head_idx != 0: # and head_idx != idx: why should we not include self-loops?
                edges.append({
                    'source': head_idx,
                    'target': idx,
                    'dep': dep,
                })
            elif head_idx == 0:
                # Optionally add edges from root node (id=0)
                edges.append({
                    'source': 0,
                    'target': idx,
                    'dep': dep,
                })

    # Check nodes are of same length as space separated words in sentence
    if len(nodes) != len(sentence.split()):
        raise Exception(f"Warning: Number of nodes ({len(nodes)}) does not match number of words in sentence ({len(sentence.split())}), sentence: {sentence}")

    return {
        'nodes': nodes,
        'edges': edges,
    }