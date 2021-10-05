import torch
from configparser import ConfigParser

def get_optimizer(model, config):
    optimizer=config['training']['optimizer']

    if optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr)
    elif optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    elif optimizer =='SGD':
        optimizer = torch.optim.SGD(model.parameters(), config.getfloat('h_param','lr'),momentum=0.9, weight_decay=0.0005)
    return optimizer