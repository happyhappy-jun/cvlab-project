import torch
import torch.nn as nn
import torch.nn.functional as F
import model.resnet
import model.wide_resnet 
import model.densenet
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

def build_model(cfg):               
    if cfg.model.type == "resnet":
        return model.resnet.resnet152()
    if cfg.model.type == "wide-resnet":
        return model.wide_resnet.Wide_ResNet(depth = 40, widen_factor=14, dropout_rate=0.3, num_classes=100)
    if cfg.model.type == "densenet":
        return model.densenet.DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=100)
    if cfg.model.type == "EfficientNet-B0":
        return EfficientNet.from_pretrained('efficientnet-b0', num_classes=100, batch_norm_momentum=0.9)