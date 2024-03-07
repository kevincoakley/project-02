import torch

#
# ResNet 18, 34, 50, 101, and 152 in TensorFlow 2
#
# He, Kaiming, et al. "Deep residual learning for image recognition." (2016) [1]
#  - https://arxiv.org/abs/1512.03385
#


class ResNetBasicBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(self, in_planes, planes, stride=1, conv_shortcut=False):
        super(ResNetBasicBlock, self).__init__()
        # Using the [3x3 , 3x3] x n convention. Section 4.1 [1]
        # Between stacks, the subsampling is performed by convolutions with
        #   a stride of 2. Section 4.1 [1]
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)

        #
        # conv_shortcut = False:
        #   The identity shortcuts (Eqn.(1)) can be directly used when the input
        #   and output are of the same dimensions (solid line shortcuts in Fig. 3).
        # conv_shortcut = True:
        #   When the dimensions increase (dotted line shortcuts in Fig. 3), we consider
        #   two options: ... (B) The projection shortcut in Eqn.(2) is used to match
        #   dimensions (done by 1×1 convolutions).
        # Section 3.3 "Residual Network" [1]
        #
        self.conv_shortcut = False
        if conv_shortcut:
            self.conv_shortcut = True
            self.shortcut = torch.nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False
            )
            self.bn_shortcut = torch.nn.BatchNorm2d(planes)

    def forward(self, inputs):
        # Following ResNet building block from Figure 2 [1].
        # We adopt batch normalization right after each convolution and before
        # activation. Section 3.4 [1]
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.conv_shortcut:
            # The projection shortcut in Eqn.(2) is used to match dimensions
            # (done by 1×1 convolutions). Section 3.3 "Residual Network" [1]
            y = self.shortcut(inputs)
            y = self.bn_shortcut(y)
        else:
            # The identity shortcuts (Eqn.(1)) can be directly used when the input
            # and output are of the same dimensions. Section 3.3 "Residual Network" [1]
            y = inputs

        out = x + y
        out = self.relu(out)

        return out


class ResNetCompleteBlock(torch.nn.Module):
    expansion: int = 4

    def __init__(self, in_planes, planes, stride=1, conv_shortcut=False):
        super(ResNetCompleteBlock, self).__init__()
        # Using the [1x1 , 3x3, 1x1] x n convention. Section 4.1 [1]
        self.conv1 = torch.nn.Conv2d(
            #in_planes, planes, kernel_size=1, stride=stride, padding=1, bias=False # KC HERE
            in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU()
        # Between stacks, the subsampling is performed by convolutions with
        # a stride of 2. Section 4.1 [1]
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)
        
        self.conv3 = torch.nn.Conv2d(
            #planes, self.expansion * planes, kernel_size=1, stride=1, padding=0, bias=False # KC HERE
            planes, self.expansion * planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = torch.nn.BatchNorm2d(self.expansion * planes)
        #
        # conv_shortcut = False:
        #   The identity shortcuts (Eqn.(1)) can be directly used when the input
        #   and output are of the same dimensions (solid line shortcuts in Fig. 3).
        # conv_shortcut = True:
        #   When the dimensions increase (dotted line shortcuts in Fig. 3), we consider
        #   two options: ... (B) The projection shortcut in Eqn.(2) is used to match
        #   dimensions (done by 1×1 convolutions).
        # Section 3.3 "Residual Network" [1]
        #
        self.conv_shortcut = False
        if conv_shortcut:
            self.conv_shortcut = True
            self.shortcut = torch.nn.Conv2d(
                in_planes, self.expansion * planes, kernel_size=1, stride=stride, padding=0, bias=False
            )
            self.bn_shortcut = torch.nn.BatchNorm2d(self.expansion * planes)

    def forward(self, inputs):
        # Following ResNet building block from Figure 2 [1].
        # We adopt batch normalization right after each convolution and before
        # activation. Section 3.4 [1]
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.conv_shortcut:
            # The projection shortcut in Eqn.(2) is used to match dimensions
            # (done by 1×1 convolutions). Section 3.3 "Residual Network" [1]
            y = self.shortcut(inputs)
            y = self.bn_shortcut(y)
        else:
            # The identity shortcuts (Eqn.(1)) can be directly used when the input
            # and output are of the same dimensions. Section 3.3 "Residual Network" [1]
            y = inputs

        out = x + y
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, basic=False):
        super(ResNet, self).__init__()
        #
        # resnet_basic_block: 2x 3x3 convolutions (ResNet18 and ResNet34)
        # resnet_complete_block: 1x 1x1 convolution, 1x 3x3 convolution,
        #   1x 1x1 convolution (ResNet50, ResNet101, and ResNet152)
        #
        if basic:
            resnet_block = ResNetBasicBlock
        else:
            resnet_block = ResNetCompleteBlock

        self.in_planes = 64

        # The first layer is 7×7 convolutions. Section 4.1 [1]
        self.zero_pad1 = torch.nn.ZeroPad2d(padding=(3, 3, 3, 3))

        self.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=0, bias=False
        )
        # We adopt batch normalization right after each convolution and before
        # activation. Section 3.4 [1]
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()

        self.zero_pad2 = torch.nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        # The numbers of filters are {64, 128, 256, 512} respectively for resnet_basic_block.
        # The numbers of filters are {256, 512, 1024, 2048} respectively for resnet_complete_block.
        # Section 4.1 [1]

        # The first stack uses 64 filters. Section 4.1 [1]
        # First block of the first stack uses identity shortcut (strides=1) since the
        #   input size matches the output size of the first layer is 3×3 convolutions.
        #   Section 3.3 "Residual Network" [1]
        self.conv2 = self._make_layer(resnet_block, 64, num_blocks[0], first_stride=1, conv_shortcut=True)
        # The second stack uses 128 filters. Section 4.1 [1]
        # First block of the second stack uses projection shortcut (strides=2)
        #   Section 3.3 "Residual Network" [1]
        self.conv3 = self._make_layer(resnet_block, 128, num_blocks[1], first_stride=2, conv_shortcut=True)
        # The third stack uses 256 filters. Section 4.1 [1]
        # First block of the third stack uses projection shortcut (strides=2)
        #   Section 3.3 "Residual Network" [1]
        self.conv4 = self._make_layer(resnet_block, 256, num_blocks[2], first_stride=2, conv_shortcut=True)
        # The forth stack uses 512 filters. Section 4.1 [1]
        # First block of the forth stack uses projection shortcut (strides=2)
        #   Section 3.3 "Residual Network" [1]
        self.conv5 = self._make_layer(resnet_block, 512, num_blocks[3], first_stride=2, conv_shortcut=True)
        # The network ends with a global average pooling, a fully-connected
        # layer, and softmax. Section 4.2 [1]
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(resnet_block.expansion * 512, num_classes)

        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, block, planes, num_blocks, first_stride, conv_shortcut):
        layers = []

        # First block of the first stack uses identity shortcut (strides=1) since the
        # First block of the second & third stack uses projection shortcut (strides=2)
        layers.append(block(self.in_planes, planes, first_stride, conv_shortcut))
        self.in_planes = block.expansion * planes

        for blocks in range(num_blocks - 1):
            # All other blocks use identity shortcut (strides=1)
            layers.append(block(self.in_planes, planes, 1, False))

        return torch.nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.zero_pad1(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.zero_pad2(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


def resnet18(num_classes=10):
    return ResNet((2, 2, 2, 2), num_classes=num_classes, basic=True)


def resnet34(num_classes=10):
    return ResNet((3, 4, 6, 3), num_classes=num_classes, basic=True)


def resnet50(num_classes=10):
    return ResNet((3, 4, 6, 3), num_classes=num_classes, basic=False)


def resnet101(num_classes=10):
    return ResNet((3, 4, 23, 3), num_classes=num_classes, basic=False)


def resnet152(num_classes=10):
    return ResNet((3, 8, 36, 3), num_classes=num_classes, basic=False)


if __name__ == "__main__":
    from torchsummary import summary

    model = resnet50()
    summary(model, (3, 224, 224))

    from torchview import draw_graph

    batch_size = 128
    model_graph = draw_graph(
        model, input_size=(batch_size, 3, 224, 224), save_graph=True, device="meta"
    )
