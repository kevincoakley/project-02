import torch
import torchvision

#
# DenseNet 121, 169, 201 & 264 for PyTorch
#
# Huang, Gao, et al. "Densely connected convolutional networks." (2017) [1]
#  - https://arxiv.org/pdf/1608.06993.pdf
#
# Code from paper:
#  https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua [2]
#


class DenseNetBasicBlock(torch.nn.Module):
    def __init__(self, in_planes, growth_rate, bottleneck):
        super(DenseNetBasicBlock, self).__init__()

        self.bottleneck = bottleneck

        post_bottleneck_in_planes = in_planes

        # A 1×1 convolution can be introduced as bottleneck layer before each 3×3 convolution
        # to reduce the number of input feature-maps, and thus to improve computational
        # efficiency. Section 3 "Bottleneck layers" [1]
        if self.bottleneck:
            self.bn_bnk = torch.nn.BatchNorm2d(in_planes)
            self.relu_bnk = torch.nn.ReLU()

            # We let each 1×1 convolution produce 4k (4 * growth_rate) feature-maps.
            # Section 3 "Bottleneck layers" [1]
            post_bottleneck_in_planes = 4 * growth_rate
            self.conv2_bnk = torch.nn.Conv2d(
                in_planes,
                post_bottleneck_in_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

        # For convolutional layers with kernel size 3×3, each side of the inputs is zero-padded by one pixel
        # to keep the feature-map size fixed. Section 3 "Implementation Details" [1]
        self.bn = torch.nn.BatchNorm2d(post_bottleneck_in_planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            post_bottleneck_in_planes,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, inputs):
        if self.bottleneck:
            x = self.bn_bnk(inputs)
            x = self.relu_bnk(x)
            x = self.conv2_bnk(x)
        else:
            x = inputs

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        out = torch.cat([inputs, x], 1)

        return out


class DenseNetTransitionBlock(torch.nn.Module):
    def __init__(self, in_planes, compression_reduction):
        super(DenseNetTransitionBlock, self).__init__()

        #
        # We use 1×1 convolution followed by 2×2 average pooling as transition layers
        # between two contiguous dense blocks. Section 3 "Implementation Details" [1]
        #
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            in_planes,
            int(in_planes * compression_reduction),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.avgpool = torch.nn.AvgPool2d(2, stride=2)

    def forward(self, inputs):
        x = self.bn1(inputs)
        x = self.relu(x)
        x = self.conv2(x)

        out = self.avgpool(x)

        return out


class DenseNet(torch.nn.Module):
    def __init__(self, num_block, num_classes=10):
        super(DenseNet, self).__init__()

        # See Section 3 "Implementation Details" [1]
        input_filter = 64
        growth_rate = 32
        compression_reduction = 0.5 
        bottleneck = True 

        # Track the growth after each dense block
        self.n_channels = input_filter

        # The first layer is 7×7 convolutions. Table 1 [1]
        self.zero_pad1 = torch.nn.ZeroPad2d(padding=(3, 3, 3, 3))

        self.conv1 = torch.nn.Conv2d(
            3, input_filter, kernel_size=7, stride=2, padding=0, bias=False
        )

        self.bn1 = torch.nn.BatchNorm2d(input_filter)
        self.relu = torch.nn.ReLU()

        self.zero_pad2 = torch.nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        #
        # DenseNet used in our experiments has four dense blocks and three transitions.
        # See Table 1 [1] for the number of layers in each dense block.
        #

        # First dense block
        self.conv2 = self._make_layer(DenseNetBasicBlock, num_block[0], growth_rate, bottleneck)

        # First transition block
        self.trans1 = DenseNetTransitionBlock(self.n_channels, compression_reduction)
        # To further improve model compactness, we can reduce the number of feature-maps at
        # transition layers. When θ = 1, the number of feature-maps across transition layers
        # remains unchanged. We refer the DenseNet with θ < 1 as DenseNet-C, and we set θ = 0.5
        # in our experiment. Section 3 "Compression" [1]
        self.n_channels = int(self.n_channels * compression_reduction)

        # Second dense block
        self.conv3 = self._make_layer(DenseNetBasicBlock, num_block[1], growth_rate, bottleneck)

        # Second transition block
        self.trans2 = DenseNetTransitionBlock(self.n_channels, compression_reduction)
        # To further improve model compactness, we can reduce the number of feature-maps at
        # transition layers. When θ = 1, the number of feature-maps across transition layers
        # remains unchanged. We refer the DenseNet with θ < 1 as DenseNet-C, and we set θ = 0.5
        # in our experiment. Section 3 "Compression" [1]
        self.n_channels = int(self.n_channels * compression_reduction)

        # Third dense block
        self.conv4 = self._make_layer(DenseNetBasicBlock, num_block[2], growth_rate, bottleneck)

        # Third transition block
        self.trans3 = DenseNetTransitionBlock(self.n_channels, compression_reduction)
        # To further improve model compactness, we can reduce the number of feature-maps at
        # transition layers. When θ = 1, the number of feature-maps across transition layers
        # remains unchanged. We refer the DenseNet with θ < 1 as DenseNet-C, and we set θ = 0.5
        # in our experiment. Section 3 "Compression" [1]
        self.n_channels = int(self.n_channels * compression_reduction)

        # Forth dense block
        self.conv5 = self._make_layer(DenseNetBasicBlock, num_block[3], growth_rate, bottleneck)

        # Last transition block before the classifier. Only use BN-ReLU after the last
        # dense block. See addTransition() [2]
        self.bn2 = torch.nn.BatchNorm2d(self.n_channels)
        #self.relu = torch.nn.ReLU()

        # At the end of the last dense block, a global average pooling is performed and
        # then a softmax classifier is attached. Section 3 "Implementation Details" [1]
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(self.n_channels, num_classes)

        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, block, num_blocks, growth_rate, bottleneck):
        layers = []

        for blocks in range(num_blocks):
            layers.append(block(self.n_channels, growth_rate, bottleneck))
            # Each layer adds k feature-maps of its own to this state. The growth rate
            # regulates how much new information each layer contributes to the global state.
            # Section 3 "Growth rate" [1]
            self.n_channels += growth_rate

        return torch.nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.zero_pad1(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.zero_pad2(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.trans1(x)

        x = self.conv3(x)
        x = self.trans2(x)

        x = self.conv4(x)
        x = self.trans3(x)

        x = self.conv5(x)

        x = self.bn2(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


def densenet121(num_classes=10):
    return DenseNet(num_block=(6, 12, 24, 16), num_classes=num_classes)

def densenet169(num_classes=10):
    return DenseNet(num_block=(6, 12, 32, 32), num_classes=num_classes)

def densenet201(num_classes=10):
    return DenseNet(num_block=(6, 12, 48, 32), num_classes=num_classes)

def densenet264(num_classes=10):
    return DenseNet(num_block=(6, 12, 64, 48), num_classes=num_classes)

if __name__ == "__main__":
    from torchsummary import summary

    model = densenet121()
    summary(model, (3, 224, 224))

    from torchview import draw_graph

    batch_size = 128
    model_graph = draw_graph(
        model, input_size=(batch_size, 3, 224, 224), save_graph=True, device="meta"
    )
