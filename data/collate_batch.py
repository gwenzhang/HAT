# encoding: utf-8

import torch


def train_collate_fn(batch):
    imgs, pids, _, _ = zip(*batch)

    # imgs, pids, c3, c4 = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    # attribute = torch.cat(attribute,dim=-1).permute(3,2,0,1)[:,0,:,:]
    # attribute = torch.tensor(attribute, dtype=torch.int64)
    # attribute = torch.tensor(attribute, dtype=torch.float)
    # attribute = []
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    # attribute = torch.tensor(attribute, dtype=torch.float)
    # attribute = []
    # attribute = torch.cat(attribute, dim=-1).permute(3, 2, 0, 1)[:,0,:,:]
    return torch.stack(imgs, dim=0), pids, camids
