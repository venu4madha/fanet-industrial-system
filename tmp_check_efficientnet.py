import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
import torch

base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
x = torch.rand(1, 3, 224, 224)
print('start', x.shape)
i = 0
for m in base.features:
    x = m(x)
    print(i, type(m).__name__, x.shape)
    i += 1
