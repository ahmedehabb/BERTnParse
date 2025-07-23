# This file contains code for another experiments done in finetuning bert alone.
# All of the models here use morpological and syntactic features, so you should uncomment the code in the training 
# script to use them.

# All work here is unreported in the paper, but you can use it for further experiments.


import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import BertModel, BertPreTrainedModel, AutoConfig
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch
from transformers import default_data_collator
from attach_morph_features import morph_features
from transformers import TrainingArguments, Trainer
from utils import coral_decode, regression_decode

# # Try Coral Bert Ordinal 
# This is a simple implementation of a BERT model with CORAL loss for ordinal regression tasks.
class BertCoralOrdinal(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_labels - 1)
        )

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)  # shape: [batch_size, num_labels - 1]

        if labels is not None:
            # Convert labels to CORAL ordinal levels (shape: [batch_size, num_labels - 1])
            levels = levels_from_labelbatch(labels, self.num_labels).to(logits.device)
            loss = coral_loss(logits, levels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}




# # Try Coral Bert With Features
# This model combines BERT with additional syntactic features and uses CORAL loss for ordinal regression
class BertCoralOrdinalWithFeatures(BertPreTrainedModel):
    def __init__(self, config, num_labels, feature_dim):
        super().__init__(config)
        self.num_labels = num_labels
        self.feature_dim = feature_dim
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Combine BERT + syntactic features
        combined_dim = config.hidden_size + feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_labels - 1)
        )

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, syntactic_feats=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = self.dropout(outputs.pooler_output)
        
        if syntactic_feats is not None:
            # [batch_size, feature_dim]
            combined = torch.cat([pooled_output, syntactic_feats], dim=1)  # Concatenate along feature axis
        else:
            print("no syntactic_feats")
            combined = pooled_output
            
        logits = self.classifier(combined)

        if labels is not None:
            # Convert labels to CORAL ordinal levels (shape: [batch_size, num_labels - 1])
            levels = levels_from_labelbatch(labels, self.num_labels).to(logits.device)
            loss = coral_loss(logits, levels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}



# # Try Coral Bert With Features and All Morph's embeddings 
class BertWithMorpholoEmbeddings(BertPreTrainedModel):
    def __init__(self, config, num_labels, morph_vocab_sizes, feature_dim, morph_emb_dim=16):
        super().__init__(config)
        self.num_labels = num_labels
        self.feature_dim = feature_dim
        self.morph_emb_dim = morph_emb_dim
        self.bert = BertModel(config)

        # Create a separate embedding layer for each morphological feature
        self.morph_embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size, morph_emb_dim)
            for feat, vocab_size in morph_vocab_sizes.items()
        })

        # Total morph embedding dimension = morph_emb_dim * number of morph features
        self.total_morph_dim = morph_emb_dim * len(morph_vocab_sizes)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        combined_dim = config.hidden_size + self.total_morph_dim + feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_labels - 1)
        )

        self.init_weights()

    def init_weights(self):
        # Call init_weights only if using transformers.BertPreTrainedModel's pattern
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                syntactic_feats=None,
                labels=None,
                ud_ids=None,
                prc3_ids=None,
                prc2_ids=None,
                prc1_ids=None,
                prc0_ids=None,
                enc0_ids=None,
                gen_ids=None,
                num_ids=None,
                cas_ids=None,
                per_ids=None,
                asp_ids=None,
                vox_ids=None,
                mod_ids=None,
                stt_ids=None,
                rat_ids=None,
                token_type_ids=None):

        # Prepare a dict with all morph inputs
        morph_inputs = {
            "ud_ids": ud_ids,
            "prc3_ids": prc3_ids,
            "prc2_ids": prc2_ids,
            "prc1_ids": prc1_ids,
            "prc0_ids": prc0_ids,
            "enc0_ids": enc0_ids,
            "gen_ids": gen_ids,
            "num_ids": num_ids,
            "cas_ids": cas_ids,
            "per_ids": per_ids,
            "asp_ids": asp_ids,
            "vox_ids": vox_ids,
            "mod_ids": mod_ids,
            "stt_ids": stt_ids,
            "rat_ids": rat_ids,
            "token_type_ids": token_type_ids,
        }

        # Filter out None values (if some features are optional)
        morph_inputs = {k: v for k, v in morph_inputs.items() if v is not None}

        # Pass BERT inputs (exclude morph features)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
    
        morph_embeds = []
        for feat_name, embed_layer in self.morph_embeddings.items():
            key = feat_name + "_ids"
            if key not in morph_inputs:
                raise ValueError(f"Missing {key} in input to forward()")
            feat_ids = morph_inputs[key]
            morph_embeds.append(embed_layer(feat_ids))  # [batch, seq_len, morph_emb_dim]

        combined_feats = torch.cat(morph_embeds, dim=-1)  # [batch, seq_len, morph_emb_dim * num_feats]
    
        # Combine BERT embeddings with morph embeddings
        combined = torch.cat([token_embeddings, combined_feats], dim=-1)
    
        # Masked mean pooling
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(combined.size()).float()
        summed = torch.sum(combined * attention_mask_expanded, dim=1)
        summed_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        pooled = summed / summed_mask  # [batch_size, combined_dim]
    
        pooled = self.dropout(pooled)
    
        # Add syntactic_feats if provided
        if syntactic_feats is not None:
            combined = torch.cat([pooled, syntactic_feats], dim=1)
        else:
            combined = pooled
    
        logits = self.classifier(combined)
    
        if labels is not None:
            levels = levels_from_labelbatch(labels, self.num_labels).to(logits.device)
            loss = coral_loss(logits, levels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}


# morph_vocab_sizes = {feat: len(vocab) for feat, vocab in feature_vocabs.items()}
# feature_dim = len(feature_columns)  # e.g., if you're using handcrafted syntactic features

# config = AutoConfig.from_pretrained(model_ckpt)

# model = BertWithMorpholoEmbeddings.from_pretrained(
#     model_ckpt,
#     config=config,
#     num_labels=num_labels,
#     feature_dim=feature_dim,
#     morph_vocab_sizes=morph_vocab_sizes,
#     morph_emb_dim=16  # or any other dimension you want
# )


# # Try Regression Bert
# This model is a simple regression model using BERT as the base, with a linear layer
# for regression output. It uses SmoothL1Loss for regression tasks.
class BertRegression(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1)
        )

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output).squeeze(-1)  # shape: [batch_size]

        if labels is not None:
          labels = labels.float()  # Important for regression
          loss = nn.SmoothL1Loss()(logits, labels) # SmoothL1Loss try too
          return {"loss": loss, "logits": logits}
        else:
          return {"logits": logits}




# # Try Regression Bert With Features
# This model combines BERT with additional syntactic features for regression tasks.
# It uses a linear layer for regression output and SmoothL1Loss for loss calculation.
class BertWithFeaturesRegression(BertPreTrainedModel):
    def __init__(self, config, num_labels, feature_dim):
        super().__init__(config)
        self.num_labels = num_labels
        self.feature_dim = feature_dim

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Combine BERT + syntactic features
        combined_dim = config.hidden_size + feature_dim
        # print("config.hidden_size", config.hidden_size)
        # print("feature dim ", feature_dim)
        # print("combined_dim ", combined_dim)
        

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1)
        )

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, syntactic_feats=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = self.dropout(outputs.pooler_output)  # [batch_size, hidden_size] = [batch_size, 768]

        if syntactic_feats is not None:
            # [batch_size, feature_dim]
            combined = torch.cat([pooled_output, syntactic_feats], dim=1)  # Concatenate along feature axis
        else:
            print("no syntactic_feats")
            combined = pooled_output

        logits = self.classifier(combined).squeeze(-1)  # [batch_size]

        if labels is not None:
            labels = labels.float()
            loss = nn.SmoothL1Loss()(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}


# Custom data collator for handling additional features, especially syntactic and morphological features.
def custom_data_collator(features):
    # Use HuggingFace's default collator to handle input_ids, attention_mask, labels, etc.
    batch = default_data_collator(features)

    # Stack syntactic features manually
    if "syntactic_feats" in features[0]:
        batch["syntactic_feats"] = torch.stack([f["syntactic_feats"] for f in features])

    # Handle morph features like ud_ids, gen_ids, etc.
    for feat in morph_features:
        key = f"{feat}_ids"
        if key in features[0]:
            batch[key] = torch.stack([f[key] for f in features])
    return batch


# # Example usage of the Trainer API with the custom model and data collator


# training_args = TrainingArguments(
#     output_dir="./barec_model",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     logging_dir="./logs",
#     per_device_train_batch_size=128,
#     per_device_eval_batch_size=128,
#     num_train_epochs=3,
#     learning_rate=5e-5,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     gradient_accumulation_steps=1,   # simulate bigger batch if needed
#     dataloader_num_workers=4,        # speed up data loading
#     fp16=True,                       # enable mixed precision for speed/memory
# )

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     # 1. Classification
#     # preds = np.argmax(logits, axis=1)
#     # 2. Coral
#     preds = coral_decode(logits)  # decode ordinal logits into class preds
#     # 3. Regression
#     # preds = regression_decode(logits, labels)

#     acc = (preds == labels).mean()
#     qwk = cohen_kappa_score(labels, preds, weights='quadratic')
#     return {"accuracy": acc, "qwk": qwk}

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=encoded_dataset,
#     eval_dataset=encoded_eval_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
#     data_collator=custom_data_collator,
# )

# trainer.train()
