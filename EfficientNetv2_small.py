import torch.nn as nn
import math
import torch
from torchvision import models
import timm


class Efficientnetv2(nn.Module):
    # def __init__(self, pretrained='./checkpoint/tf_efficientnetv2_b3-57773f13.pth'):
    def __init__(self, pretrained='./checkpoint/tf_efficientnetv2_s-eb54923e.pth'):
    # def __init__(self, pretrained='./checkpoint/tf_efficientnetv2_s_21ft1k-d7dafa41.pth'):
        super().__init__()
        self.rgb_encoder: nn.Module = timm.create_model(
            model_name="tf_efficientnetv2_s", features_only=True, out_indices=range(1, 5)
        )
        # self.depth_encoder: nn.Module = timm.create_model(
        #     model_name="tf_mobilenetv3_small_075", features_only=True, out_indices=range(1, 5)
        # )
        if pretrained:
            self.rgb_encoder.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False)
            # self.depth_encoder.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False)

    def forward(self, data):
        rgb_feats = self.rgb_encoder(data)
        # to cnn decoder for fusion
        x1 = rgb_feats[0]
        x2 = rgb_feats[1]
        x3 = rgb_feats[2]
        x4 = rgb_feats[3]

        return x1, x2, x3, x4

