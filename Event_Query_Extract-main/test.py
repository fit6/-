import torch
file = './data/DuEE1.0/pt/train.pt'
tr_dataset = torch.load(file)
print(len(tr_dataset))