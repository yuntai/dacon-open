import torch
from torch.utils.data.sampler import WeightedRandomSampler


def calc_weights(targets):
    cnt = targets.unique(return_counts=True)[1]
    return cnt.sum()/cnt

targets = torch.load("targets.pt")
w_per_class = calc_weights(targets)
w_per_samples = [w_per_class[t] for t in targets]
w_sampler = WeightedRandomSampler(w_per_samples, 1000)
ds = torch.utils.data.TensorDataset(targets)
dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, sampler=w_sampler)

batches = []
for batch, in dl:
    batches.append(batch)

batches = torch.cat(batches)
x = batches.unique(return_counts=True)[1]
print(x, x.sum(), x/x.sum(), (x/x.sum()).mean(), (x/x.sum()).std())
