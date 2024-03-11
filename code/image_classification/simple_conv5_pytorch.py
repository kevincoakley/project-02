import torch

# Simple Convolutional Neural Network

class SimpleBasicBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(SimpleBasicBlock, self).__init__()
        # Between stacks, the subsampling is performed by convolutions with
        #   a stride of 2. 
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU()
        

    def forward(self, inputs):
        # We adopt batch normalization right after each convolution and before
        # activation. 
        x = self.conv1(inputs)
        x = self.bn1(x)
        out = self.relu(x)
        return out


class Simple(torch.nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(Simple, self).__init__()

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

        # The first stack uses 64 filters. 
        self.conv2 = self._make_layer(SimpleBasicBlock, 64, num_blocks, first_stride=1)
        # The second stack uses 128 filters.
        self.conv3 = self._make_layer(SimpleBasicBlock, 128, num_blocks, first_stride=2)
        # The third stack uses 256 filters. 
        self.conv4 = self._make_layer(SimpleBasicBlock, 256, num_blocks, first_stride=2)
        # The forth stack uses 512 filters. 
        self.conv5 = self._make_layer(SimpleBasicBlock, 512, num_blocks, first_stride=2)

        # The network ends with a global average pooling, a fully-connected
        # layer, and softmax. 
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, block, planes, num_blocks, first_stride):
        layers = []

        # First block of the first stack uses identity shortcut (strides=1) since the
        # First block of the second & third stack uses projection shortcut (strides=2)
        layers.append(block(self.in_planes, planes, first_stride))
        self.in_planes = planes

        for blocks in range(num_blocks - 1):
            # All other blocks use identity shortcut (strides=1)
            layers.append(block(self.in_planes, planes, 1))

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


def simple5_1(num_classes=10):
    return Simple(1, num_classes=num_classes)

def simple5_3(num_classes=10):
    return Simple(3, num_classes=num_classes)

def simple5_5(num_classes=10):
    return Simple(5, num_classes=num_classes)

def simple5_7(num_classes=10):
    return Simple(7, num_classes=num_classes)

def simple5_9(num_classes=10):
    return Simple(9, num_classes=num_classes)

def simple5_11(num_classes=10):
    return Simple(11, num_classes=num_classes)

def simple5_13(num_classes=10):
    return Simple(13, num_classes=num_classes)

def simple5_15(num_classes=10):
    return Simple(15, num_classes=num_classes)

if __name__ == "__main__":
    from torchsummary import summary

    model = simple5_1()
    summary(model, (3, 224, 224))

    from torchview import draw_graph

    batch_size = 128
    model_graph = draw_graph(
        model, input_size=(batch_size, 3, 224, 224), save_graph=True, device="meta"
    )
