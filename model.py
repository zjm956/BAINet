import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from ResNet import ResNet50
from Res2Net import Res2Net50
from modules import BEM, EIA, ConvCBR
from modules import HFEM as EnhanceBlock
from EfficientNetv2_small import Efficientnetv2
from TripletAttention import TripletAttention


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config

        if config.backbone == 'resnet':
            self.resnet = ResNet50()
        elif config.backbone == 'res2net':
            self.res2net = Res2Net50()
        elif config.backbone == 'efficientnet':
            self.efficientnet = Efficientnetv2()

        self.enhance5 = EnhanceBlock(256, 64)
        self.enhance4 = EnhanceBlock(160, 64)
        self.enhance3 = EnhanceBlock(64, 64)
        self.enhance2 = EnhanceBlock(48, 64)

        self.bem = BEM()

        self.eia1 = EIA(64, 64, 64)
        self.eia2 = EIA(64, 64, 64)
        self.eia3 = EIA(64, 64, 64)

        self.CBR1 = ConvCBR(64, 64, kernel_size=3, padding=1)
        self.CBR2 = ConvCBR(64, 64, kernel_size=3, padding=1)
        self.CBR3 = ConvCBR(64, 64, kernel_size=3, padding=1)

        self.upsample3_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.out = nn.Conv2d(64, 1, 1)
        self.out1 = nn.Conv2d(48, 1, 1)
        self.out2 = nn.Conv2d(64, 1, 1)
        self.out3 = nn.Conv2d(160, 1, 1)
        self.out4 = nn.Conv2d(256, 1, 1)

        # self._initialize_weight()

    def forward(self, x):
        target_size = x.shape[2:]
        x1, x2, x3, x4, = self.efficientnet(x)

        o_x1 = self.out1(x1)
        o_x2 = self.out2(x2)
        o_x3 = self.out3(x3)
        o_x4 = self.out4(x4)

        x4 = self.enhance5(x4)
        x3 = self.enhance4(x3)
        x2 = self.enhance3(x2)
        x1 = self.enhance2(x1)

        e_x1 = self.out(x1)
        e_x2 = self.out(x2)
        e_x3 = self.out(x3)
        e_x4 = self.out(x4)

        edge, out_edge = self.bem(x1, x2, x4)
        edge_att = torch.sigmoid(out_edge)

        rf3, rp3, bf3, bp3 = self.eia3(x3, x4, edge)
        rf2, rp2, bf2, bp2 = self.eia2(x2, rf3, bf3)
        rf1, rp1, bf1, bp1 = self.eia1(x1, rf2, bf2)

        out_rf3 = self.upsample3_2(self.CBR3(rf3))
        out_rf2 = self.upsample2_1(self.CBR2(out_rf3 + rf2))
        out_rf1 = self.CBR1(out_rf2 + rf1)
        out_cam = self.out(out_rf1)

        out_e = F.interpolate(edge_att, size=target_size, mode='bilinear', align_corners=False)
        cam = F.interpolate(out_cam, size=target_size, mode='bilinear', align_corners=False)
        out_rp3 = F.interpolate(rp3, size=target_size, mode='bilinear', align_corners=False)
        out_bp3 = F.interpolate(bp3, size=target_size, mode='bilinear', align_corners=False)
        out_rp2 = F.interpolate(rp2, size=target_size, mode='bilinear', align_corners=False)
        out_bp2 = F.interpolate(bp2, size=target_size, mode='bilinear', align_corners=False)
        out_rp1 = F.interpolate(rp1, size=target_size, mode='bilinear', align_corners=False)
        out_bp1 = F.interpolate(bp1, size=target_size, mode='bilinear', align_corners=False)
        out_x1 = F.interpolate(o_x1, size=target_size, mode='bilinear', align_corners=False)
        out_x2 = F.interpolate(o_x2, size=target_size, mode='bilinear', align_corners=False)
        out_x3 = F.interpolate(o_x3, size=target_size, mode='bilinear', align_corners=False)
        out_x4 = F.interpolate(o_x4, size=target_size, mode='bilinear', align_corners=False)
        out_ex1 = F.interpolate(e_x1, size=target_size, mode='bilinear', align_corners=False)
        out_ex2 = F.interpolate(e_x2, size=target_size, mode='bilinear', align_corners=False)
        out_ex3 = F.interpolate(e_x3, size=target_size, mode='bilinear', align_corners=False)
        out_ex4 = F.interpolate(e_x4, size=target_size, mode='bilinear', align_corners=False)

        # return out_e, out_rp3, out_bp3, out_rp2, out_bp2, out_rp1, out_bp1
        # return cam, out_rp3, out_rp2, out_rp1, out_e, out_bp3, out_bp2, out_bp1, out_x1, out_x2, out_x3, out_x4
        return cam, out_rp3, out_rp2, out_rp1, out_e, out_bp3, out_bp2, out_bp1, out_x1, out_x2, out_x3, out_x4, out_ex1, out_ex2, out_ex3, out_ex4
        # return cam, out_rp3, out_rp2, out_rp1, out_e, out_bp3, out_bp2, out_bp1

        # return [torch.sigmoid(upsample(x1_rp, target_size)), torch.sigmoid(upsample(x2_rp, target_size)),
        #         torch.sigmoid(upsample(x3_rp, target_size)), torch.sigmoid(upsample(x4_rp, target_size))], \
        #        [torch.sigmoid(upsample(x1_bp, target_size)), torch.sigmoid(upsample(x2_bp, target_size)),
        #         torch.sigmoid(upsample(x3_bp, target_size)), torch.sigmoid(upsample(x4_bp, target_size))]

    def _initialize_weight(self):
        if self.config.backbone == 'resnet':
            res50 = models.resnet50(pretrained=True)
            pretrained_dict = res50.state_dict()
        else:
            # pretrained_dict = torch.load('./save_path/model_BCNet_v3/Net_epoch_60.pth')
            pretrained_dict = torch.load('./checkpoint/res2net50_v1b_26w_4s-3cf99910.pth')

        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)



if __name__ == '__main__':
    from options import opt
    model = Net(opt)
    img = torch.randn((3, 3, 256, 256))
    out = model(img)