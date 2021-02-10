import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models.resnet import wide_resnet50_2
from torchvision.models import resnet50
from model import resnet
from  model import wide_resnet 
from model import densenet
from efficientnet_pytorch import EfficientNet


# class Net_arch(nn.Module):
#     # Network architecture
#     def __init__(self, cfg):
#         super(Net_arch, self).__init__()
#         self.cfg = cfg

#         # TODO: This is example code. You should change this part as you need
#         self.lrelu = nn.LeakyReLU()
#         self.conv1 = nn.Sequential(nn.Conv2d(1, 4, 3, 2, 1), self.lrelu)
#         self.conv2 = nn.Sequential(nn.Conv2d(4, 4, 3, 2, 1), self.lrelu)
#         self.fc = nn.Linear(7 * 7 * 4, 10)

#     def forward(self, x):  # x: (B,1,28,28)
#         # TODO: This is example code. You should change this part as you need
#         x = self.conv1(x)  # x: (B,4,14,14)
#         x = self.conv2(x)  # x: (B,4,7,7)
#         x = torch.flatten(x, 1)  # x: (B,4*7*7)
#         x = self.fc(x)  # x: (B,10)
#         return x

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def build_model(cfg):               
    if cfg.model.type == "resnet":
        model_ft = resnet50(pretrained=True)
        set_parameter_requires_grad(model_ft, False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 100)
        return model_ft
    if cfg.model.type == "wide-resnet":
        return wide_resnet50_2(pretrained=True, num_classes = 100)
    if cfg.model.type == "densenet":
        return densenet.DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=100)
    if cfg.model.type == "EfficientNet-B0":
        return EfficientNet.from_pretrained('efficientnet-b0', num_classes=100, batch_norm_momentum=0.9)