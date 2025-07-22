import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def coral_decode(logits):
    """
    logits: [batch_size, num_classes-1] tensor
    returns: numpy array of predicted class labels
    """
    probas = torch.sigmoid(logits)  # sigmoid over logits
    # For each sample, count how many thresholds are > 0.5
    preds = (probas > 0.5).sum(dim=1)
    return preds
    
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            logits = out["logits"]

            # for coral
            preds = coral_decode(logits)

            # CE decode
            # preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return {"accuracy": acc, "qwk": qwk}

def predict(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            logits = out["logits"]
            
            # CE decode
            # preds = logits.argmax(dim=1)
            
            # coral decode
            preds = coral_decode(logits)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    return all_preds, all_labels

def output_to_file(model, loader):
    model.eval()
    all_preds, all_ids = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # dummy: set real labels all = 0
            if hasattr(data, 'y'):
                data.y = torch.zeros_like(data.y, device=data.y.device, dtype=torch.long)

            out = model(data)
            logits = out["logits"]

            # Use coral decoding or CE depending on your training
            preds = coral_decode(logits)
            preds = [int(pred) + 1 for pred in preds]
            all_preds.extend(preds)
            all_ids.extend(data.id.cpu().numpy())  # assumes sentence_id exists

    # Save to file named exactly "prediction" (no .csv)
    with open("prediction", "w") as f:
        f.write("Sentence ID,Prediction\n")
        for sid, pred in zip(all_ids, all_preds):
            f.write(f"{sid},{pred}\n")

    return all_preds, all_ids