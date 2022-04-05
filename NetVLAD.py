import torch
import torch.nn as nn
import torch.nn.functional as F
#

#CNN(VGG16)에서 마지막 conv layer의 출력인 H*W*D 의 output을 descriptor들의 집합으로 본다.
class NetVLAD(nn.Module):
    def __init__(self, num_clusters=12, dim=128, alpha=100.0, normalize_input=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1,1), bias=False)
        self.centroids = nn.Parameter(torch. rand(num_clusters, dim))
        self._init_params()
    
    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
    
    def forward(self, x): 
        N, C = x.shape[:2] #cnn의 결과 : W*H*D 에서 W랑 D?
        
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        #a 계수의 soft assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad