# Defining our models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import NNConv, GCNConv, global_mean_pool, GATv2Conv, NNConv, TransformerConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn import TransformerConv, GATConv, GraphNorm
from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pad_sequence
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom loss function for ordinal regression
# This is a weighted kappa loss function that is used for ordinal regression tasks
# It computes the weighted kappa statistic between predicted and true labels
# The weights can be linear or quadratic, and the loss is computed as log(1 - kappa)
# This loss is useful for tasks where the classes have an ordinal relationship, such as sentiment analysis or grading systems
# The implementation is based on the paper: "Weighted kappa loss function for multi-class classification of ordinal data in deep learning"
class WeightedKappaLoss(nn.Module):
    def __init__(self, num_classes, mode='quadratic', eps=1e-10):
        super().__init__()
        self.num_classes = num_classes
        assert mode in ['linear', 'quadratic'], "Mode must be 'linear' or 'quadratic'"
        self.mode = mode
        self.eps = eps

        # Create weight matrix omega
        labels = torch.arange(num_classes, dtype=torch.float32)
        diff = labels.unsqueeze(0) - labels.unsqueeze(1)  # shape (C,C)
        if mode == 'linear':
            weights = diff.abs()
        else:
            weights = diff.pow(2)

        # Normalize weights by max to keep in [0,1]
        weights /= weights.max()

        # Register as buffer for device compatibility
        self.register_buffer('weights', weights)

    def forward(self, y_pred, y_true):
        """
        y_pred: Float tensor of shape (N, C), probabilities (softmax outputs)
        y_true: Long tensor of shape (N,), true class indices
        """

        device = y_pred.device
        N = y_pred.size(0)
        C = self.num_classes

        # One-hot encode y_true: shape (N, C)
        y_true_onehot = torch.zeros((N, C), device=device)
        y_true_onehot.scatter_(1, y_true.unsqueeze(1), 1)

        # Normalize y_pred probabilities to sum to 1 (if not already)
        y_pred = y_pred / (y_pred.sum(dim=1, keepdim=True) + self.eps)

        # Observed matrix O_{i,j} = sum over samples of predicted prob class i and true class j
        O = y_pred.T @ y_true_onehot  # shape (C, C)

        # Marginals for predictions and truth
        hist_pred = y_pred.sum(dim=0)  # shape (C,)
        hist_true = y_true_onehot.sum(dim=0)  # shape (C,)

        # Expected matrix E_{i,j} = outer product of marginals normalized by number of samples
        E = torch.outer(hist_pred, hist_true) / N  # shape (C, C)

        # Weighted sums
        weighted_O = (self.weights * O).sum()
        weighted_E = (self.weights * E).sum()

        # Weighted kappa
        kappa = 1 - (weighted_O / (weighted_E + self.eps))

        # Loss is log(1 - kappa) as per paper
        # from paper: 
        # However, optimization of the loss function (L ) is normally presented as a minimization problem, therefore in Eq. (4) we present
        # the same problem converted to minimization. Notice that in this case, we propose to take the logarithm of the index, in order to increase the penalization of incorrect assignments.
        # minimize L = log (1 − κ ) where L ∈ (−∞, log 2] (4)
        loss = torch.log(1 - kappa + self.eps)
        return loss



# Simple GNN model with edge features and positional embeddings
# This model uses a simple edge aggregation mechanism and attention pooling over graph nodes
# It also uses a simple edge projection and node projection architecture
# This model is a good baseline for many tasks and should perform well on simple datasets
class ImprovedSimpleEdgeGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_relations, num_pos_tags, pos_emb_dim=63, dropout=0.1, conv_layers=3):
        super().__init__()
        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.wkl = WeightedKappaLoss(num_classes)
        self.pos_emb = nn.Embedding(num_pos_tags, pos_emb_dim)

        edge_embedding_size = hidden_dim
        self.edge_emb = nn.Embedding(num_relations, edge_embedding_size)
        # Try larger too ?
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_embedding_size, edge_embedding_size),
            nn.ReLU(),
            nn.Linear(edge_embedding_size, edge_embedding_size)
        )

        input_dim_total = input_dim + pos_emb_dim
        self.node_proj = nn.Sequential(
            nn.Linear(input_dim_total, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_fuse_proj = nn.Linear(edge_embedding_size, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.conv_layers = conv_layers
        self.dropout = nn.Dropout(dropout)
        num_heads = 4
        for _ in range(conv_layers):
            self.convs.append(TransformerConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=num_heads, concat=False, edge_dim=edge_embedding_size))  # replace with your conv class
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.att_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )

        self.fc = nn.Linear(hidden_dim, num_classes - 1)
        self.fc_wkl = nn.Linear(hidden_dim, num_classes)

    def aggregate_edge_features(self, x, edge_index, edge_features):
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
    
        # Init aggregation tensors
        src_counts = torch.zeros(x.size(0), device=x.device)
        agg_edge_feats_src = torch.zeros(x.size(0), edge_features.size(1), device=x.device)
        src_counts.index_add_(0, src_nodes, torch.ones_like(src_nodes, dtype=torch.float))
        agg_edge_feats_src.index_add_(0, src_nodes, edge_features)
    
        dst_counts = torch.zeros(x.size(0), device=x.device)
        agg_edge_feats_dst = torch.zeros(x.size(0), edge_features.size(1), device=x.device)
        dst_counts.index_add_(0, dst_nodes, torch.ones_like(dst_nodes, dtype=torch.float))
        agg_edge_feats_dst.index_add_(0, dst_nodes, edge_features)
    
        # Normalize
        src_counts = src_counts.clamp(min=1).unsqueeze(-1)
        dst_counts = dst_counts.clamp(min=1).unsqueeze(-1)
        agg_edge_feats_src = agg_edge_feats_src / src_counts
        agg_edge_feats_dst = agg_edge_feats_dst / dst_counts
    
        # Fuse
        agg_edge_feats = torch.cat([agg_edge_feats_src, agg_edge_feats_dst], dim=-1)
        agg_edge_feats = self.edge_fuse_proj(agg_edge_feats)
        return agg_edge_feats
    
        
    def forward(self, data):
        x, y = data.x, data.y
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        pos_tag_ids = data.pos_tag_ids  # [N, 4]
        mask = (pos_tag_ids != 0).unsqueeze(-1)  # [N, 4, 1]
        
        pos_emb = self.pos_emb(pos_tag_ids)  # [N, 4, 32]
        pos_emb = pos_emb * mask.float()     # zero out padding embeddings
        
        # Compute masked average
        sum_emb = pos_emb.sum(dim=1)  # [N, 32]
        valid_counts = mask.sum(dim=1).clamp(min=1)  # avoid division by zero → [N, 1]
        pos_features = sum_emb / valid_counts  # [N, 32]

        edge_features = self.edge_emb(edge_attr.squeeze())  # [num_edges, edge_dim]
        edge_features = self.edge_proj(edge_features)      # project to hidden_dim back
        # edge_features = F.relu(self.edge_proj(edge_features))  # project to hidden_dim back and relu
        
        x = torch.cat([x, pos_features], dim=-1) # add pos to x
        x = self.node_proj(x)
        x = F.relu(x)
        x_res = x # save x to use it in the very end 
        
        x_prev = x
        for i in range(self.conv_layers):
            x_new = self.convs[i](x_prev, edge_index, edge_features)
            x_new = self.norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            
            # Residual from previous layer and edge aggregation
            x = x_prev + x_new
            x = x + self.aggregate_edge_features(x, edge_index, edge_features)
            
            x_prev = x  # update for next iteration
            
        x = x + x_res  # original input residual

                
        x_pool = self.att_pool(x, batch)

        # Improves stability before the final classifier. This layer doesn’t add trainable weights — it just normalizes.
        x_pool = F.layer_norm(x_pool, x_pool.size()[1:])
        logits = self.fc(x_pool)
        
        # coral loss
        levels = levels_from_labelbatch(y, self.num_classes).to(logits.device)
        loss = coral_loss(logits, levels)

        # wkl loss
        logits_wkl = self.fc_wkl(x_pool)
        probabilities = torch.softmax(logits_wkl, dim=1)
        wkl_loss = self.wkl(probabilities, y)
        
        loss = 0.5 * loss + 0.5 * wkl_loss

        # predict from coral head, since it provided better results in general
        return {"loss": loss, "logits": logits}



# Another GNN model with more complex edge aggregation and attention pooling
# This model uses a gating mechanism to aggregate edge features and multi-head attention pooling over graph nodes
# It also uses a more complex edge projection and node projection architecture
# This model is more powerful and should perform better on complex datasets
class StrongerEdgeGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_relations, num_pos_tags,
                 pos_emb_dim=63, dropout=0.1, conv_layers=4, num_heads=4):
        super().__init__()
        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.wkl = WeightedKappaLoss(num_classes)
        self.pos_emb = nn.Embedding(num_pos_tags, pos_emb_dim)

        self.edge_emb = nn.Embedding(num_relations, hidden_dim)
        self.edge_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        input_dim_total = input_dim + pos_emb_dim
        self.node_proj = nn.Sequential(
            nn.Linear(input_dim_total, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge gating mechanism
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.edge_fuse_proj = nn.Linear(hidden_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(conv_layers):
            if i % 2 == 0:
                self.convs.append(TransformerConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, edge_dim=hidden_dim))
            else:
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, edge_dim=hidden_dim))    
            self.norms.append(GraphNorm(hidden_dim))

        # Multi-head attention pooling over graph nodes
        self.global_query = nn.Parameter(torch.randn(1, hidden_dim))
        self.att_pool = MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_classes - 1)
        self.fc_wkl = nn.Linear(hidden_dim, num_classes)

    def aggregate_edge_features(self, x, edge_index, edge_features):
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        x_src = x[src_nodes]
        x_dst = x[dst_nodes]

        # Gating
        gate_input = torch.cat([x_src, x_dst, edge_features], dim=-1)
        gates = self.edge_gate(gate_input)
        gated_edge_feats = gates * edge_features

        # Aggregate
        agg_edge_feats_src = torch.zeros(x.size(0), edge_features.size(1), device=x.device)
        agg_edge_feats_src.index_add_(0, src_nodes, gated_edge_feats)

        agg_edge_feats_dst = torch.zeros(x.size(0), edge_features.size(1), device=x.device)
        agg_edge_feats_dst.index_add_(0, dst_nodes, gated_edge_feats)

        agg = (agg_edge_feats_src + agg_edge_feats_dst) / 2
        return self.edge_fuse_proj(agg)

    def forward(self, data):
        x, y = data.x, data.y
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        pos_tag_ids = data.pos_tag_ids

        mask = (pos_tag_ids != 0).unsqueeze(-1)
        pos_emb = self.pos_emb(pos_tag_ids) * mask.float()
        pos_features = pos_emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        edge_features = self.edge_proj(self.edge_emb(edge_attr.squeeze()))

        x = torch.cat([x, pos_features], dim=-1)
        x = F.relu(self.node_proj(x))
        x_res = x

        x_prev = x
        for i in range(self.conv_layers):
            if isinstance(self.convs[i], TransformerConv):
                x_new = self.convs[i](x_prev, edge_index, edge_features)
            else:
                x_new = self.convs[i](x_prev, edge_index)

            x_new = self.norms[i](x_new, batch)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)

            x = x_prev + x_new + self.aggregate_edge_features(x_new, edge_index, edge_features)
            x_prev = x

        x = x + x_res

        # Attention pooling: batch graph → [batch_size, num_nodes, hidden_dim]        
        # Gather node embeddings per graph
        graphs = []
        for i in range(batch.max().item() + 1):
            node_indices = (batch == i).nonzero(as_tuple=True)[0]
            graphs.append(x[node_indices])
        
        # Pad graphs to same length
        padded = pad_sequence(graphs, batch_first=True)   # [B, N_max, H]
        mask = torch.ones(padded.size()[:2], dtype=torch.bool, device=x.device)
        for i, g in enumerate(graphs):
            mask[i, len(g):] = False
        
        # Attention pooling
        query = self.global_query.expand(padded.size(0), 1, -1)  # [B, 1, H]
        pooled, _ = self.att_pool(query, padded, padded, key_padding_mask=~mask)
        x_pool = pooled.squeeze(1)

        x_pool = F.layer_norm(x_pool, x_pool.size()[1:])
        logits = self.fc(x_pool)

        levels = levels_from_labelbatch(y, self.num_classes).to(logits.device)
        loss = coral_loss(logits, levels)

        logits_wkl = self.fc_wkl(x_pool)
        probabilities = torch.softmax(logits_wkl, dim=1)
        wkl_loss = self.wkl(probabilities, y)

        loss = 0.2 * loss + 0.8 * wkl_loss
        return {"loss": loss, "logits": logits}



# BERT-based model that combines BERT embeddings with a GNN
# This model uses BERT to generate word embeddings and then applies a GNN on top of those embeddings
# It tokenizes sentences, extracts word embeddings, and constructs a graph representation
# The GNN processes the graph and outputs class logits

class BERTGraphModel(nn.Module):
    def __init__(self, gnn_model, tokenizer, bert_model):
        super().__init__()
        self.gnn_model = gnn_model
        self.tokenizer = tokenizer
        self.bert_model = bert_model

    def forward(self, batch):
        # batch is a PyG Batch (collection of Data objects with sentence, etc.)
        batch_size = batch.num_graphs
        sentences = batch.sentence  # List[str], len == batch_size
        batch_words = batch.sentence_tokenized

        # Step 1: Tokenize sentences
        encoding = self.tokenizer(
            batch_words,
            is_split_into_words=True,
            return_tensors='pt',
            add_special_tokens=True,
            padding=True,
            truncation=True,
        )
        word_ids_list = [encoding.word_ids(batch_index=i) for i in range(batch_size)]

        # Move to device
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        # with torch.no_grad():
        # outputs = self.bert_model(**encoding)
        # hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden]
        
        outputs = self.bert_model(**encoding, output_hidden_states=True)
        hidden_states_all = outputs.hidden_states  # tuple of (13,) [embedding + 12 hidden layers]
        
        # Example: sum last 4 layers (you can also learn weights)
        last_4_layers = hidden_states_all[-4:]  # list of tensors [B, seq_len, hidden_dim]
        hidden_states = torch.stack(last_4_layers).mean(dim=0)  # average last 4 layers
        
        # Step 2: Convert BERT embeddings to word-level x features
        x_list = []
        for i in range(batch_size):
            words = batch_words[i]
            word_ids = word_ids_list[i]
            token_embeds = hidden_states[i]
            num_words = len(batch.pos_tag_ids[i]) - 1  # exclude ROOT

            word_embeddings = []
            for j, word in enumerate(words):
                indices = [k for k, wid in enumerate(word_ids) if wid == j]
                if indices:
                    vecs = token_embeds[indices]
                    word_emb = vecs.mean(dim=0)
                else:
                    word_emb = torch.zeros(token_embeds.size(-1), device=token_embeds.device)
                word_embeddings.append(word_emb)
            
            # Use the [CLS] token embedding as root node for the sentence
            cls_emb = token_embeds[0]  # [CLS] is always at position 0 in BERT output
            word_embeddings = [cls_emb] + word_embeddings

            x_graph = torch.stack(word_embeddings, dim=0)  # [N_nodes, input_dim]
            x_list.append(x_graph)

        # Step 3: Concatenate and inject into the PyG Batch
        x_full = torch.cat(x_list, dim=0)  # [total_nodes_across_batch, input_dim]
        batch.x = x_full

        # Step 4: Pass to GNN
        return self.gnn_model(batch)