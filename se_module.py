import torch
import torch.nn.functional as F
import math
from torch import nn

from torch.nn.parameter import Parameter


class GhostSEModule(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=1, ratio=2, dw_size=3):
        super(GhostSEModule, self).__init__()
        self.ghost_module = GhostModule(channel, channel, kernel_size, ratio, dw_size)
        self.se_layer = SELayer(channel, reduction)

    def forward(self, x):
        x = self.ghost_module(x)
        x = self.se_layer(x)
        return x


class GhostECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2, ratio=2, dw_size=3):
        super(GhostECA, self).__init__()
        # 根据通道数求出卷积核的大小kernel_size
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ghost_module = GhostModule(channel, channel, kernel_size=kernel_size, ratio=ratio, dw_size=dw_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ghost_module(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# class GhostECA(nn.Module):
#     def __init__(self, channel, b=1, gamma=2, ratio=2):
#         super(GhostECA, self).__init__()
#         kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
#         kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.ghost = GhostModule(channel, channel // ratio)
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)  # 通道权重计算
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.ghost(y)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)


# class GhostModule(nn.Module):
#     def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
#         super(GhostModule, self).__init__()
#         # ratio一般会指定成2，保证输出特征层的通道数等于exp
#         self.oup = oup
#         init_channels = math.ceil(oup / ratio)
#         new_channels = init_channels * (ratio - 1)
#
#         # 利用1x1卷积对输入进来的特征图进行通道的浓缩，获得特征通缩
#         # 跨通道的特征提取
#         self.primary_conv = nn.Sequential(
#             nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
#             # 1x1卷积的输入通道数为GhostModule的输出通道数oup/2
#             nn.BatchNorm2d(init_channels),  # 1x1卷积后进行标准化
#             nn.ReLU(inplace=True) if relu else nn.Sequential(),  # ReLU激活函数
#         )
#
#         # 在获得特征浓缩后，使用逐层卷积，获得额外的特征图
#         # 跨特征点的特征提取    一般会设定大于1的卷积核大小
#         self.cheap_operation = nn.Sequential(
#             nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
#             # groups参数的功能就是将普通卷积转换成逐层卷据
#             nn.BatchNorm2d(new_channels),
#             nn.ReLU(inplace=True) if relu else nn.Sequential(),
#         )
#
#     def forward(self, x):
#         x1 = self.primary_conv(x)
#         x2 = self.cheap_operation(x1)
#         # 将1x1卷积后的结果和逐层卷积后的结果进行堆叠
#         out = torch.cat([x1, x2], dim=1)
#         return out[:, :self.oup, :, :]


class GhostModule(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, ratio=2, dw_size=3, relu=True):
        super(GhostModule, self).__init__()
        self.output_channels = output_channels  # 设定输出通道数，即最终的通道数
        init_channels = output_channels // ratio  # 计算Ghost模块中主干部分的通道数
        new_channels = init_channels * (ratio - 1)  # 计算Ghost模块中影子部分的通道数

        # 主干部分，进行普通的卷积操作
        self.primary_conv = nn.Sequential(
            nn.Conv2d(input_channels, init_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()  # 是否使用ReLU激活函数取决于relu参数
        )

        # 影子部分，进行depthwise卷积操作
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, stride=1, padding=(dw_size - 1) // 2, groups=init_channels,
                      bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()  # 是否使用ReLU激活函数取决于relu参数
        )

    def forward(self, x):
        x1 = self.primary_conv(x)  # 主干部分的卷积操作，得到主干部分的输出
        x2 = self.cheap_operation(x1)  # 影子部分的depthwise卷积操作，得到影子部分的输出
        out = torch.cat([x1, x2], dim=1)  # 沿着通道维度进行拼接，将主干部分和影子部分合并
        return out[:, :self.output_channels, :, :]  # 返回合并后的张量，并截取前面设定的输出通道数


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        assert channel > reduction, "Make sure your input channel bigger than reduction which equals to {}".format(
            reduction)
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP  除以16是降维系数
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)  # kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 结果相加
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 声明卷积核为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 进行相应的same padding填充
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        # 拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 7x7卷积填充为3，输入通道为2，输出通道为1
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        # ----------------------------------#
        # 根据通道数求出卷积核的大小kernel_size
        # ----------------------------------#
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ------------------------------------------#
        # 显示全局平均池化,再是k*k的卷积,
        # 最后为Sigmoid激活函数,进而得到每个通道的权重w
        # 最后进行回承操作,得出最终结果
        # ------------------------------------------#
        y = self.avg_pool(x)
        # y.squeeze(-1)是将最后一个维度删掉即宽这个维度就没有了，transpose(-1, -2)是将最后一个和倒数第二个维度进行互换，即现在的维度变成了b，1，c这三个维度，1是由于前面的自适应平均层变成了1*1的图像，所以长在这里就是1。unsqueeze(-1)是增加最后一个维度
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()

        t = torch.cat((channel_mean, channel_std), dim=2)
        return t

    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)

        return x * g


class GELayer(nn.Module):
    def __init__(self, channel, layer_idx):
        super(GELayer, self).__init__()

        # Kernel size w.r.t each layer for global depth-wise convolution
        kernel_size = [-1, 56, 28, 14, 7][layer_idx]

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=kernel_size, groups=channel),
            nn.BatchNorm2d(channel),
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        gate = self.conv(x)
        gate = self.activation(gate)
        return x * gate


class SoftPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(SoftPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x

    def soft_pool2d(self, x, kernel_size=2, stride=None, force_inplace=False):
        kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        else:
            stride = (stride, stride)
        _, c, h, w = x.shape
        e_x = torch.sum(torch.exp(x), dim=1, keepdim=True)
        return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(
            F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))

#
def soft_pool2d(x, kernel_size=2, stride=None):
    kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = (stride, stride)
    _, c, h, w = x.shape
    e_x = torch.exp(x)
    return F.avg_pool2d(x * e_x, kernel_size, stride=stride) * (sum(kernel_size)) / (
            F.avg_pool2d(e_x, kernel_size, stride=stride) * (sum(kernel_size)))


class SoftPool2D(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(SoftPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x


class mixedPool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0, alpha=0.5):
        # nn.Module.__init__(self)
        super(mixedPool, self).__init__()
        alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha)  # nn.Parameter is special Variable
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.alpha * F.max_pool2d(x, self.kernel_size, self.stride, self.padding) + (
                1 - self.alpha) * F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x
