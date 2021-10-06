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