import torch


def wide_resnet_lr(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return init * pow(0.2, optim_factor)

def get_scheduler(cfg, optimizer):
    model_type = cfg.model.type
    optimizer_mode = cfg.model.optimizer
    total_epochs = cfg.num_epoch
    if model_type == "wide-resnet" and optimizer_mode == "SGD":
        lambda1 = lambda epoch: wide_resnet_lr(cfg.train.optimizer.SGD.lr, epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    elif model_type == "resnet" and optimizer_mode == "SGD":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cfg.num_epoch.total)
    elif model_type == "densenet" and optimizer_mode == "SAM":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [total_epochs*0.3, total_epochs*0.6, total_epochs*0.8], gamma = 0.2)
    elif model_type == "densenet" and optimizer_mode == "SGD":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [total_epochs*0.5, total_epochs*0.75], gamma = 0.1)
    elif model_type == "EfficientNet-B0" and optimizer_mode == "SGD":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [40, 60, 80], gamma = 0.1)
    else:
        scheduler = None
    return scheduler
