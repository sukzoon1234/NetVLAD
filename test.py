import torch
import torch.nn as nn
import torch.nn.functional as F
#
num_clusters = num_clusters
dim = dim
alpha = alpha
normalize_input = normalize_input
conv = nn.conv2d(dim, num_clusters, kernel_size=(1,1), bias=False)
centroids = nn.Parameter(torch. rand(num_clusters, dim))_init_params()

self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
    