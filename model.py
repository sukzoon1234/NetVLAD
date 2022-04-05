import torch
import torch.nn as nn
from torchvision.models import resnet18,vgg16

from NetVLAD import NetVLAD

class VPR_model(nn.Module):
    def __init__(self, num_clusters=16, dim=512):
        super(VPR_model, self).__init__()

        self.num_clusters = num_clusters
        self.dim = dim
        self.encoder = vgg16(pretrained=True)
        self.layers = list(self.encoder.features.children())[:-2]
        self._init_params()
        self.encoder = nn.Sequential(*(self.layers))
        self.pool = NetVLAD(self.num_clusters, self.dim)

    def _init_params(self):
        for l in self.layers[:-5]:  #conv1부터 conv4까지만 weight 학습, conv5 뒤로는 고정시키기
            for p in l.parameters():
                p.requires_grad = False


    def forward(self, x): 
        descriptor = self.encoder(x) #out = W*H*D(512)
        vlad = self.pool(descriptor)
        return vlad