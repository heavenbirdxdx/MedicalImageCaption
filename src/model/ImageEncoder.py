from torchvision.models import resnet50
import torch
import torch.nn as nn

# This is a Encoder model consisting of ResNet50

class ImageEncoder(nn.Module):
    def __init__(self, cnn, encoded_image_size=14):
        super().__init__()

        # cnn = resnet50()
        self.image_model = cnn
        # modules = list(self.image_model.children())[:-2]
        # self.image_model = nn.Sequential(*modules)
        del self.image_model.fc
        self.image_model.fc = torch.nn.Linear(2048, 256)

        # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.cnn = self.image_model
    
    def encode(self, image_matrix):
        image_feature = self.cnn(image_matrix)
        # image_feature = self.adaptive_pool(image_feature)
        # image_feature = image_feature.permute(0, 2, 3, 1)

        return image_feature

    def forward(self, image_matrix):

        image_feature = self.encode(image_matrix)

        return image_feature