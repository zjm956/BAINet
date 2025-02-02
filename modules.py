import torch
import time
import numbers
import einops
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from TripletAttention import TripletAttention


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvCBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvCBR, self).__init__()
        self.conv = SeparableConv2d(in_channel, out_channel,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvCBR_init(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvCBR_init, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ECA(nn.Module):
    """ Efficient Channel Attention (ECA); https://arxiv.org/abs/1910.03151 (CVPR2019)
        code implementation from :
        https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # (B,C,H,W) -> (B,C,1,1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # (B,C,1,1) -> (B,1,C) -> (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        # element-wise multiplication (residual connection)
        return x * y.expand_as(x)


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=1):
#         super(ChannelAttention, self).__init__()
#
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = max_out
#         return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

'''
Efficient MSCA
'''
# class EMA(nn.Module):
#     def __init__(self, channels, c2=None, factor=32):
#         super(EMA, self).__init__()
#         self.groups = factor
#         assert channels // self.groups > 0
#         self.softmax = nn.Softmax(-1)
#         self.agp = nn.AdaptiveAvgPool2d((1, 1))
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
#         self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
#         self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
#         x_h = self.pool_h(group_x)
#         x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
#         hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
#         x_h, x_w = torch.split(hw, [h, w], dim=2)
#         x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
#         x2 = self.conv3x3(group_x)
#         x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
#         return (group_x * weights.sigmoid()).reshape(b, c, h, w)


'''
ASBI_BEM
'''
# class PA(nn.Module):
#     def __init__(self, in_dim):
#         super(PA, self).__init__()
#         # Posting-H
#         self.query_conv_h = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
#         self.key_conv_h = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
#         self.value_conv_h = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
#         # Posting-W
#         self.query_conv_w = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
#         self.key_conv_w = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
#         self.value_conv_w = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 1))
#         self.la = nn.Parameter(torch.zeros(1))
#         self.lb = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)
#         # finally refine
#         self.conv_final = nn.Conv2d(2 * in_dim, in_dim,
#                                     kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         B, C, H, W = x.size()
#         axis_h = 1
#         axis_h *= H
#         view = (B, -1, axis_h)
#         projected_query_h = self.query_conv_h(x).view(*view).permute(0, 2, 1)  # [B,H,CW]
#         projected_key_h = self.key_conv_h(x).view(*view)  # [B,CW,H]
#         attention_map_h = torch.bmm(projected_query_h, projected_key_h)  # [B,H,H]
#         attention_h = self.softmax(attention_map_h)  # [B,H,H]
#         projected_value_h = self.value_conv_h(x).view(*view)  # [B,WC,H]
#         out_h = torch.bmm(projected_value_h, attention_h.permute(0, 2, 1))  # [B,wc,h]
#         out_h = out_h.view(B, C, H, W)  # [b, c, h, w]
#         out_h = self.la * out_h
#         # Position-W
#         axis_w = 1
#         axis_w *= W
#         view = (B, -1, axis_w)
#         projected_query_w = self.query_conv_w(x).view(*view).permute(0, 2, 1)  # [B,H,CW]
#         projected_key_w = self.key_conv_w(x).view(*view)  # [B,CW,H]
#         attention_map_w = torch.bmm(projected_query_w, projected_key_w)  # [B,H,H]
#         attention_w = self.softmax(attention_map_w)  # [B,H,H]
#         projected_value_w = self.value_conv_w(x).view(*view)  # [B,WC,H]
#         out_w = torch.bmm(projected_value_w, attention_w.permute(0, 2, 1))  # [B,wc,h]
#         out_w = out_w.view(B, C, H, W)  # [b, c, h, w]
#         out_w = self.lb * out_w
#         out_fianl = torch.cat([out_h, out_w], 1)
#         out_final = self.conv_final(out_fianl) + x
#         return out_final

# class CALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y

# Residual Channel Attention Block (RCAB)
# class RCAB(nn.Module):
#     def __init__(
#             self, n_feat, kernel_size=3, reduction=16,
#             bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#
#         super(RCAB, self).__init__()
#         modules_body = []
#         for i in range(2):
#             modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
#             if bn: modules_body.append(nn.BatchNorm2d(n_feat))
#             if i == 0: modules_body.append(act)
#         modules_body.append(CALayer(n_feat, reduction))
#         self.body = nn.Sequential(*modules_body)
#         self.res_scale = res_scale
#
#     def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
#         return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
#
#     def forward(self, x):
#         res = self.body(x)
#         # res = self.body(x).mul(self.res_scale)
#         res += x
#         return res

class MFAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFAM, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_1_1 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_2 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_3 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_4 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_5 = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv_5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x1 = self.conv_1_1(x)
        x2 = self.conv_1_2(x)
        x3 = self.conv_1_3(x)
        x_3_1 = self.relu(self.conv_3_1(x2))  ## (BS, 32, ***, ***)
        x_5_1 = self.relu(self.conv_5_1(x3))  ## (BS, 32, ***, ***)
        x_3_2 = self.relu(self.conv_3_2(x_3_1 + x_5_1))  ## (BS, 64, ***, ***)
        x_5_2 = self.relu(self.conv_5_2(x_5_1 + x_3_1))  ## (BS, 64, ***, ***)
        x_mul = torch.mul(x_3_2, x_5_2)
        out = self.relu(x1 + self.conv_1_5(x_mul + x_3_1 + x_5_1))

        return out


'''
ASPP
'''
class ASPP_simple(nn.Module):
    def __init__(self, inplanes, planes, rates=[1, 6, 12, 18]):
        super(ASPP_simple, self).__init__()

        self.aspp0 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1,
                                                 stride=1, padding=0, dilation=1, bias=False),
                                       nn.BatchNorm2d(planes))
        self.aspp1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                                 stride=1, padding=rates[1], dilation=rates[1], bias=False),
                                       nn.BatchNorm2d(planes))
        self.aspp2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                                 stride=1, padding=rates[2], dilation=rates[2], bias=False),
                                       nn.BatchNorm2d(planes))
        self.aspp3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                                 stride=1, padding=rates[3], dilation=rates[3], bias=False),
                                       nn.BatchNorm2d(planes))

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))

        self.reduce = nn.Sequential(
                nn.Conv2d(planes * 5, planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.global_avg_pool(x)
        x4 = F.upsample(x4, x3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x0, x1, x2, x3, x4), dim=1)
        x = self.reduce(x)
        return x

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x


class BEM(nn.Module):
    def __init__(self):
        super(BEM, self).__init__()
        self.conv_0 = ConvCBR_init(128, 64, kernel_size=3, padding=1)
        self.conv_1 = ConvCBR(64, 64, kernel_size=3, padding=1)
        self.conv_2 = ConvCBR(64, 64, kernel_size=3, padding=1)
        self.conv_3 = ConvCBR(64, 64, kernel_size=3, padding=1)
        self.conv_4 = ConvCBR(64, 64, kernel_size=3, padding=1)
        self.conv_5 = ConvCBR_init(64, 64, kernel_size=3, padding=1)
        self.conv_out = ConvCBR_init(64, 1, kernel_size=1)
        # self.pa = PA(64)
        self.pa = TripletAttention(False)
        # self.ca = RCAB(64)
        self.ca = ECA(64, 1)
        self.sa = SpatialAttention()

    def forward(self, x1, x2, x4):
        if x2.size()[2:] != x1.size()[2:]:
            x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear')
        if x4.size()[2:] != x1.size()[2:]:
            x4 = F.interpolate(x4, size=x1.size()[2:], mode='bilinear')

        x12 = torch.cat((x1, x2), dim=1)
        x12 = self.conv_0(x12)
        x_ca = self.ca(x12)

        x12_5 = x_ca * self.pa(x4)

        x12_conv = self.conv_2(self.conv_1(x12_5))
        xe = x12_conv + x12_5
        xe = self.conv_3(xe)

        xe_sa = self.sa(xe) * xe

        xe_conv = self.conv_4(xe_sa)
        out = self.conv_5(xe_conv + x_ca)

        fe_out = self.conv_out(out)
        return out, fe_out

# class GM(nn.Module):
#     def __init__(self, channel):
#         super(GM, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         m_batchsize, C, height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
#
#         out1 = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out1 = out1.view(m_batchsize, C, height, width)
#
#         return F.relu(x + out1)


def upsample(tensor, size):
    return F.interpolate(tensor, size, mode='bilinear', align_corners=True)


def norm_layer(channel, norm_name='bn', _3d=False):
    if norm_name == 'bn':
        return nn.BatchNorm2d(channel) if not _3d else nn.BatchNorm3d(channel)
    elif norm_name == 'gn':
        return nn.GroupNorm(min(32, channel // 4), channel)


class HFEM(nn.Module):
    # set dilation rate = 1
    def __init__(self, in_channel, out_channel):
        super(HFEM, self).__init__()
        self.in_C = in_channel
        self.temp_C = out_channel

        # first branch
        self.head_branch1 = nn.Sequential(
            nn.Conv2d(self.in_C, self.temp_C, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )
        self.conv1_branch1 = nn.Sequential(
            nn.Conv2d(self.temp_C, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.conv2_branch1 = nn.Sequential(
            nn.Conv2d(self.temp_C, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.conv3_branch1 = nn.Sequential(
            nn.Conv2d(self.temp_C, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.tail_branch1 = nn.Conv2d(self.temp_C, self.temp_C, 3, 1, padding=2, dilation=2, bias=False)

        # second branch
        self.head_branch2 = nn.Sequential(
            nn.Conv2d(self.in_C, self.temp_C, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )
        self.conv1_branch2 = nn.Sequential(
            nn.Conv2d(self.temp_C, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.conv2_branch2 = nn.Sequential(
            nn.Conv2d(self.temp_C, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.tail_branch2 = nn.Conv2d(self.temp_C, self.temp_C, 3, 1, padding=2, dilation=2, bias=False)

        # third branch
        self.head_branch3 = nn.Sequential(
            nn.Conv2d(self.in_C, self.temp_C, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )
        self.conv1_branch3 = nn.Sequential(
            nn.Conv2d(self.temp_C, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.tail_branch3 = nn.Conv2d(self.temp_C, self.temp_C, 3, 1, padding=2, dilation=2, bias=False)

        # forth branch
        self.head_branch4 = nn.Sequential(
            nn.Conv2d(self.in_C, self.temp_C, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )
        self.tail_branch4 = nn.Conv2d(self.temp_C, self.temp_C, 3, 1, padding=2, dilation=2, bias=False)

        # convs for fusion
        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(self.temp_C * 2, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(self.temp_C * 2, self.temp_C, 3, 1, 1, bias=False),
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )
        self.fusion_conv3 = nn.Sequential(
            nn.Conv2d(self.temp_C * 2, self.temp_C, 3, 1, 1, bias=False),  # output channel = temp_C
            norm_layer(self.temp_C),
            nn.ReLU(True)
        )

        # channel attention
        # self.ca = ChannelAttention(out_channel)
        self.ca = ECA(64, 1)

    def forward(self, x):
        x_branch1_0 = self.head_branch1(x)
        x_branch1_1 = self.conv1_branch1(x_branch1_0)
        x_branch1_2 = self.conv2_branch1(x_branch1_1)
        x_branch1_3 = self.conv3_branch1(x_branch1_2)
        x_branch1_tail = self.tail_branch1(x_branch1_3)

        x_branch2_0 = self.head_branch2(x)
        x_branch2_0 = torch.cat([x_branch2_0,
                                 upsample(x_branch1_tail, x_branch2_0.shape[2:])], dim=1)
        x_branch2_0 = self.fusion_conv1(x_branch2_0)
        x_branch2_1 = self.conv1_branch2(x_branch2_0)
        x_branch2_2 = self.conv2_branch2(x_branch2_1)
        x_branch2_tail = self.tail_branch2(x_branch2_2)

        x_branch3_0 = self.head_branch3(x)
        x_branch3_0 = torch.cat([x_branch3_0,
                                 upsample(x_branch2_tail, x_branch3_0.shape[2:])], dim=1)
        x_branch3_0 = self.fusion_conv2(x_branch3_0)
        x_branch3_1 = self.conv1_branch3(x_branch3_0)
        x_branch3_tail = self.tail_branch3(x_branch3_1)

        x_branch4_0 = self.head_branch4(x)
        x_branch4_0 = torch.cat([x_branch4_0,
                                 upsample(x_branch3_tail, x_branch4_0.shape[2:])], dim=1)
        x_branch4_0 = self.fusion_conv3(x_branch4_0)
        x_branch4_tail = self.tail_branch4(x_branch4_0)

        # x_output = torch.cat([upsample(x_branch1_tail, x_branch4_tail.shape[2:]),
        #                       upsample(x_branch2_tail, x_branch4_tail.shape[2:]),
        #                       upsample(x_branch3_tail, x_branch4_tail.shape[2:]),
        #                       x_branch4_tail], dim=1)
        # x_output = self.fusion_cat(x_output)
        x_output = self.ca(x_branch4_tail) * x_branch4_tail
        return x_output


class Split(nn.Module):
    def __init__(self, channel=64):
        super(Split, self).__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            norm_layer(channel, _3d=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            norm_layer(channel, _3d=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            norm_layer(channel, _3d=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            norm_layer(channel, _3d=True),
            nn.ReLU(inplace=True),
        )
        self.edge_pred = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 3, 1, 1)
        )
        self.region_pred = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 3, 1, 1)
        )

    def forward(self, x):
        edge = self.edge_conv(x)
        x = x.unsqueeze(2)
        edge = edge.unsqueeze(2)
        x_cat = torch.cat([x, edge], dim=2)
        x_cat_out = self.fusion(x_cat)

        x, edge = x_cat_out[:, :, 0, :, :], x_cat_out[:, :, 1, :, :]

        x_pred = self.region_pred(x)
        edge_pred = self.edge_pred(edge)
        return x, x_pred, edge, edge_pred


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """
    def __init__(self, in_dims=64, token_dim=32, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        # self.final = nn.Linear(token_dim * num_heads, token_dim)
        self.final = nn.Linear(token_dim * num_heads, token_dim * num_heads)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        query = self.to_query(x)
        key = self.to_key(x)
        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        G = torch.sum(A * query, dim=1) # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD

        out = self.Proj(G * key) + query #BxNxD
        out = self.final(out) # BxNxD
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

class IntegralAttention (nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(IntegralAttention, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.conv_cat = nn.Sequential(
            BasicConv2d(in_channel*3, out_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),

        )
        self.conv_res = BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1)

        self.Aatt = EfficientAdditiveAttnetion()

        self.ConvOut = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))
        fuse = self.relu(x_cat + self.conv_res(x))

        # can change to the MS-CAM or SE Attention, refer to lib.RCAB.
        # context = (fuse.pow(2).sum((2,3), keepdim=True) + self.eps).pow(0.5) # [B, C, 1, 1]

        channel_add_term = self.Aatt(fuse)
        out = channel_add_term * fuse + fuse
        out = self.ConvOut(out)

        return out


# EIA module
class EIA(nn.Module):
    def __init__(self, s_channel = 64, h_channel= 64 ,e_channel= 64 ):
        super(EIA, self).__init__()
        #self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv_1 =  ConvCBR(h_channel, h_channel,kernel_size=3, stride=1,padding=1)
        self.conv_2 =  ConvCBR(s_channel, h_channel,kernel_size=3, stride=1,padding=1)
        self.conv_3 =  ConvCBR(e_channel, e_channel,kernel_size=3, stride=1,padding=1)
        self.conv_d1 = BasicConv2d(h_channel, h_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = BasicConv2d(h_channel, h_channel, kernel_size=3, stride=1, padding=1)

        self.attention = IntegralAttention()
        self.convfuse1 = nn.Sequential(
            BasicConv2d(s_channel*3, s_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(s_channel, s_channel,kernel_size=3, stride=1,padding=1),
        )
        self.split = Split()

    def forward(self, left, down, edge):
        left_1 = self.conv_1(left)
        down_1 = self.conv_2(down)
        edge_1 = self.conv_3(edge)

        down_2 = self.conv_d1(down_1)
        left_2 = self.conv_l(left_1)

    #z1 conv(down) * left
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode='bilinear')
        z1 = F.relu(left_1 * down_2, inplace=True)

    #z2 conv(left) * down
        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')
        z2 = F.relu(down_1 * left_2, inplace=True)

        if edge_1.size()[2:] != left.size()[2:]:
            edge_1 = F.interpolate(edge_1, size=left.size()[2:], mode='bilinear')

        fuse = self.convfuse1(torch.cat((z1, z2, edge_1), 1))
        eim_out = self.attention(fuse)
        out_rf, out_rp, out_bf, out_bp = self.split(eim_out)

        return out_rf, out_rp, out_bf, out_bp

























