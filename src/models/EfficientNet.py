import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EfficientNet(nn.Module):
    def __init__(self, num_class, model_name='b0', pretrained=True):
        super().__init__()

        self.backbone = self.getEfficientNet(model_name, pretrained)
        self.classifier = nn.Linear(1000, num_class)

    def forward(self, images):
        outputs = self.backbone(images)
        outputs = self.classifier(outputs)
        return outputs

    def getEfficientNet(self, name: str, pretrained: bool):
        if name == 'b0':
            return models.efficientnet_b0(pretrained=pretrained)
        elif name == 'b1':
            return models.efficientnet_b1(pretrained=pretrained)
        elif name == 'b2':
            return models.efficientnet_b2(pretrained=pretrained)
        elif name == 'b3':
            return models.efficientnet_b3(pretrained=pretrained)
        elif name == 'b4':
            return models.efficientnet_b4(pretrained=pretrained)
        elif name == 'b5':
            return models.efficientnet_b5(pretrained=pretrained)
        elif name == 'b6':
            return models.efficientnet_b6(pretrained=pretrained)
        elif name == 'b7':
            return models.efficientnet_b7(pretrained=pretrained)
        elif name == 'v2_l':
            return models.efficientnet_v2_l(pretrained=pretrained)
        elif name == 'v2_m':
            return models.efficientnet_v2_m(pretrained=pretrained)
        elif name == 'v2_s':
            return models.efficientnet_v2_s(pretrained=pretrained)
