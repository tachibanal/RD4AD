import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.ConvTranspose2d:
    """2x2 deconvolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if stride == 2:
            self.conv1 = deconv2x2(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = deconv2x2(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class DeResNet(nn.Module):
    def match_tensor_shapes(self, tensor_list: List[Tensor], target_tensor: Tensor) -> List[Tensor]:
        """Match all tensors to the shape of the target tensor (height, width, and channels)."""
        target_c, target_h, target_w = target_tensor.shape[1], target_tensor.shape[2], target_tensor.shape[3]
        return [
            F.interpolate(tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
            if tensor.shape[1] == target_c else
            F.interpolate(tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
            for tensor in tensor_list
        ]

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000):
        super(DeResNet, self).__init__()
        self.inplanes = 512  # Match with the encoder's last output channel
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, y: List[Tensor], res: int) -> List[Tensor]:
        # Adjust shapes of skip connection tensors
        if isinstance(y, list):
            y = self.match_tensor_shapes(y, x)

        # Perform forward pass
        if res == 0:
            feature_a = self.layer1(x)  # 512 -> 256
            feature_b = self.layer2(feature_a)  # 256 -> 128
            feature_c = self.layer3(feature_b)  # 128 -> 64
        else:
            feature_a = self.layer1(x)
            feature_b = self.layer2(self.relu(feature_a + y[2]))
            feature_c = self.layer3(self.relu(feature_b + y[1]))

        return [feature_c, feature_b, feature_a]


def de_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DeResNet:
    return DeResNet(BasicBlock, [2, 2, 2])


# Test example
if __name__ == "__main__":
    encoder_output = [
        torch.randn(1, 512, 8, 8),  # Example encoder output (layer 1)
        torch.randn(1, 256, 16, 16),  # Example encoder output (layer 2)
        torch.randn(1, 128, 32, 32),  # Example encoder output (layer 3)
    ]
    decoder = de_resnet18()
    x = torch.randn(1, 512, 8, 8)  # Example input to decoder
    outputs = decoder(x, encoder_output, res=1)
    print([o.shape for o in outputs])
