import re
from typing import Any, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from torch import Tensor

from collections import OrderedDict
from se_module import GhostECA, ECA, CBAM, SELayer, GhostModule, GCT, SRMLayer, GELayer, SoftPool2D, SoftPool2d
import torch.utils.model_zoo as model_zoo
from thop import profile


class _DenseLayer(nn.Module):
    def __init__(self,
                 input_c: int,
                 growth_rate: int,
                 bn_size: int,
                 drop_rate: float,
                 memory_efficient: bool = False):
        super(_DenseLayer, self).__init__()
        # 第一个卷积层
        # Add SELayer at here, like SE-PRE block in original paper illustrates
        self.add_module("selayer", GCT(input_c))
        self.add_module("norm1", nn.BatchNorm2d(input_c))  # 添加一个BatchNorm2d层，用于对输入进行批标准
        self.add_module("relu1", nn.LeakyReLU(inplace=True))  # 添加一个ReLU激活函数层，将其应用在输入张量上
        self.add_module("conv1", nn.Conv2d(in_channels=input_c,
                                           out_channels=bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1,
                                           bias=False))  # 添加一个1x1卷积层

        # 第二个卷积层
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.LeakyReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate,
                                           growth_rate,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False))  # 添加一个3x3卷积层

        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        # 将输入张量连接在通道维度上
        concat_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))
        return bottleneck_output

    @staticmethod
    def any_requires_grad(inputs: List[Tensor]) -> bool:
        # 检查输入张量是否有梯度要求
        for tensor in inputs:
            if tensor.requires_grad:
                return True

        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, inputs: List[Tensor]) -> Tensor:
        # 使用checkpointing执行forward函数中的一部分，以节省内存
        def closure(*inp):
            return self.bn_function(inp)

        return cp.checkpoint(closure, *inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        if isinstance(inputs, Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("memory efficient not supported in JIT")
            # 使用checkpointing执行bn_function函数
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self,
                 num_layers: int,
                 input_c: int,
                 bn_size: int,
                 growth_rate: int,
                 drop_rate: float,
                 memory_efficient: bool = False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            # 创建_DenseLayer层，并添加到模型中
            layer = _DenseLayer(input_c + i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:

        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    # 使用1*1卷积和2*2平均池化作为两个连续密集块之间的过渡层
    def __init__(self,
                 input_c: int,
                 output_c: int):
        super(_Transition, self).__init__()
        # 过渡层
        self.add_module("norm", nn.BatchNorm2d(input_c))  # 对输入进行通道维度上的标准化操
        self.add_module("relu", nn.LeakyReLU(inplace=True))  # 添加一个ReLU激活函数层
        self.add_module("conv", nn.Conv2d(input_c,
                                          output_c,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False))  # 添加一个1x1卷积层，用于调整通道数GCT
        self.add_module("pool", SoftPool2D(kernel_size=2, stride=2))  # ：添加一个平均池化层，将特征图的尺寸减少一半


class DenseNet(nn.Module):
    """
growth_rate（增长率）是指每个层中要添加的滤波器（即卷积核）的数量。它决定了每个密集块（dense block）中每个层的输出通道数。
block_config（块配置）是一个包含4个整数的列表，用于指定每个池化块（pooling block）中有多少层。例如，(6, 12, 24, 16) 表示第一个池化块有6层，第二个池化块有12层，以此类推。
num_init_features（初始特征数）是指在第一个卷积层中学习的滤波器（卷积核）的数量。它决定了输入图像经过第一个卷积层后的输出通道数。
bn_size（瓶颈层倍数）是一个乘性因子，用于确定瓶颈层中的特征映射通道数。即瓶颈层的输出通道数为 bn_size * growth_rate。
drop_rate（丢弃率）是在每个密集层（dense layer）后应用的丢弃（dropout）比率。丢弃是一种正则化技术，用于减少过拟合。
num_classes（分类类别数）是要分类的类别数量。这决定了最终全连接层的输出维度，与数据集的类别数相匹配。
memory_efficient（内存效率）是一个布尔值，表示是否使用内存效率的检查点（checkpointing）技术。当设置为 True 时，模型使用检查点技术以节省内存，但会导致计算效率稍微降低。当设置为 False 时，不使用检查点技术。
    """

    def __init__(self,
                 growth_rate: int = 32,
                 block_config: Tuple[int, int, int, int] = (6, 12, 48, 32),
                 num_init_features: int = 64,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 num_classes: int = 1000,
                 memory_efficient: bool = False):
        super(DenseNet, self).__init__()
        # 第一层卷积conv+bn+relu+pool
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.LeakyReLU(inplace=True)),
            ("pool0", SoftPool2D(kernel_size=3, stride=2, padding=1)),
        ]))

        # Add SELayer at first convolution
        # self.features.add_module("SELayer_0a", SELayer(channel=num_init_features))

        # 每个Dense Block
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):

            # Add a SELayer
            self.features.add_module("GCTLayer_%da" % (i + 1), GCT(num_features))
            # 创建_DenseBlock层，并添加到模型
            block = _DenseBlock(num_layers=num_layers,
                                input_c=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                # 创建_Transition层，并添加到模型中
                # Add a SELayer behind each transition block
                self.features.add_module("GCTLayer_%db" % (i + 1), GCT(num_features))
                trans = _Transition(input_c=num_features,
                                    output_c=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

            # 最后的Batch Normalization层
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        # 全连接层
        self.classifier = nn.Linear(num_features, num_classes)


        # 初始化模型中的权重和偏置
        for m in self.modules():
            # 卷积层使用 Kaiming 正态分布初始化方法，适用于激活函数为 ReLU
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# DenseNet121(k=32):blocks=[6,12,24,16]
# DenseNet169(k=32):blocks=[6,12,32,32]
# DenseNet201(k=32):blocks=[6,12,48,32]
# DenseNet161(k=48):blocks=[6,12,36,24]
def densenet121(**kwargs: Any) -> DenseNet:
    # Top-1 error: 25.35%
    # 'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 24, 16),
                    num_init_features=64,
                    **kwargs)


def densenet169(**kwargs: Any) -> DenseNet:
    # Top-1 error: 24.00%
    # 'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 32, 32),
                    num_init_features=64,
                    **kwargs)


def densenet201(**kwargs: Any) -> DenseNet:
    # Top-1 error: 22.80%
    # 'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 48, 32),
                    num_init_features=64,
                    **kwargs)


def densenet161(**kwargs: Any) -> DenseNet:
    # Top-1 error: 22.35%
    # 'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth'
    return DenseNet(growth_rate=48,
                    block_config=(6, 12, 36, 24),
                    num_init_features=96,
                    **kwargs)


def load_state_dict(model: nn.Module, weights_path: str) -> None:
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = torch.load(weights_path)

    num_classes = model.classifier.out_features
    load_fc = num_classes == 1000

    for key in list(state_dict.keys()):
        if load_fc is False:
            if "classifier" in key:
                del state_dict[key]

        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=load_fc)

    print("successfully load pretrain-weights.")
# se_densenet = DenseNet()
# print('==================1. 通过print打印网络结构=====================')
# print(se_densenet)   # 1. 通过print打印网络结构
#
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())
#
# 创建一个DenseNet模型
# model = densenet121()
#
# # 计算参数数量
# num_params = count_parameters(model)
# print("Total number of parameters: {}".format(num_params))