import torch
from configparser import ConfigParser

def get_optimizer(model, optimizer, lr):

    if optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    elif optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    elif optimizer =='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = lr,momentum=0.9, weight_decay=0.0005)
    return optimizer

def get_scheduler(optimizer, scheduler):
    if scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 10,
            eta_min = 0  
        )
    elif scheduler == 'LRscheduler':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                        lr_lambda=lambda epoch: 0.95 ** epoch,
                        last_epoch=-1,
                        verbose=False)

    return scheduler