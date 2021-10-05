import torch

def get_optimizer(model, config):
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr)
    elif config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    elif config.optimizer =='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr= config.lr,momentum=0.9, weight_decay=0.0005)
    return optimizer