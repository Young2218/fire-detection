import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VisionTransformer(nn.Module):
    def __init__(self, num_class, model_name='b_16', pretrained=True):
        super().__init__()

        self.backbone = self.getVit(model_name, pretrained)
        self.classifier = nn.Linear(1000, num_class)

    def forward(self, images):
        outputs = self.backbone(images)
        outputs = self.classifier(outputs)
        return outputs

    def getVit(self, name: str, pretrained: bool):
        if name == 'b_16':
            return models.vit_b_16(pretrained=pretrained)
