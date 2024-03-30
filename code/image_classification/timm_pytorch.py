import torch
import timm

class Timm(torch.nn.Module):
    def __init__(self, timm_model, input_shape=(224, 224, 3), num_classes=10, pretrained=False):
        super(Timm, self).__init__()

        self.model = timm.create_model(timm_model, pretrained=pretrained, num_classes=num_classes, img_size=input_shape[0])

    def forward(self, x):
        return self.model(x)


def timm_vit_s_8(input_shape=(224, 224, 3), num_classes=10, pretrained=False):
    return Timm(timm_model="vit_small_patch8_224", input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)

def timm_vit_b_8(input_shape=(224, 224, 3), num_classes=10, pretrained=False):
    return Timm(timm_model="vit_base_patch8_224", input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)

def timm_vit_t_16(input_shape=(224, 224, 3), num_classes=10, pretrained=False):
    return Timm(timm_model="vit_tiny_patch16_224", input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)

def timm_vit_s_16(input_shape=(224, 224, 3), num_classes=10, pretrained=False):
    return Timm(timm_model="vit_small_patch16_224", input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)

def timm_vit_b_16(input_shape=(224, 224, 3), num_classes=10, pretrained=False):
    return Timm(timm_model="vit_base_patch16_224", input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)

def timm_vit_l_16(input_shape=(224, 224, 3), num_classes=10, pretrained=False):
    return Timm(timm_model="vit_large_patch16_224", input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)

def timm_vit_h_16(input_shape=(224, 224, 3), num_classes=10, pretrained=False):
    return Timm(timm_model="vit_huge_patch14_224", input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    from torchsummary import summary

    model = timm_vit_b_16()
    summary(model, (3, 224, 224))

    from torchview import draw_graph

    batch_size = 128
    model_graph = draw_graph(
        model, input_size=(batch_size, 3, 224, 224), save_graph=True, device="meta"
    )
