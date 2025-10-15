import torch

def topk_o2m(scores, k=6):
    # scores: (Ncells, Ngt); top-k per GT → True mask
    pos = torch.zeros_like(scores, dtype=torch.bool)
    if scores.numel()==0: return pos
    topk_idx = torch.topk(scores, k=min(k, scores.shape[0]), dim=0).indices
    for j in range(scores.shape[1]):
        pos[topk_idx[:,j], j] = True
    return pos

def greedy_o2o(scores):
    # (Ncells, Ngt) → one index per GT (or -1)
    Ncells, Ngt = scores.shape
    taken = torch.zeros(Ncells, dtype=torch.bool, device=scores.device)
    match = torch.full((Ngt,), -1, dtype=torch.long, device=scores.device)
    for j in range(Ngt):
        col = scores[:, j].masked_fill(taken, -1e9)
        i = torch.argmax(col)
        if col[i] > 0: match[j] = i; taken[i] = True
    return match
