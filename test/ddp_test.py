import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
sys.path.insert(0, ".") # run with python ./test/ddp_test.py
import common

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DistributedSampler

# https://pytorch.org/docs/stable/distributed.html

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

        self.net2.weight.data.normal_(mean=0.0, std=0.01)
        #print("init:", self.net2.weight.sum())

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def calc_weights(targets):
    return 1./targets.unique(return_counts=True)[1]

def calc_prop(targets):
    cnts = targets.unique(return_counts=True)[1]
    return cnts/cnts.sum()

def setup_dataloader(targets):
    from sampler import DistributedSamplerWrapper
    w_per_class = calc_weights(targets)
    w_per_samples = [w_per_class[t] for t in targets]

    w_sampler = WeightedRandomSampler(w_per_samples, len(w_per_samples))

    ixs = torch.arange(len(targets))
    ds = torch.utils.data.TensorDataset(ixs, targets)

    sampler = DistributedSamplerWrapper(w_sampler)

    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, sampler=sampler)
    return dl

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    targets = torch.multinomial(torch.FloatTensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.5,0.9]), 1000, replacement=True)
    #targets = torch.load("targets.pt")
    if rank == 0:
        p = calc_prop(targets)
        print("prop=", p)

    dl = setup_dataloader(targets)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    #print(f"after DDP {rank}", model.net2.weight.sum())

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    batches = []
    ixs = []
    for ix, batch in dl:
        batches.append(batch)
        ixs.append(ix)

        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()

    #TODO: barrier
    batches = torch.cat(batches)
    ixs = torch.cat(ixs)
    p = calc_prop(targets[ixs])
    print(f"thru ixs {rank=}", p, p.mean(), p.std())

    report(batches, rank)

    #print(f"done {rank}", model.net2.weight.data.detach().sum())
    cleanup()

def report(batches, rank):
    p = calc_prop(batches)

    print(f"{rank=} {batches.shape=}")
    cnt = batches.unique(return_counts=True)[1].detach()
    print(f"{rank=} {cnt=} prop=", p, len(p), p.mean(), f"{p.std()*100:.2f}%")

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    targets = torch.multinomial(torch.FloatTensor([1,1,1,1,1,1,1,2,2,5]), num_samples=1000, replacement=True)
    torch.save(targets, "targets.pt")

    run_demo(demo_basic, 2)
