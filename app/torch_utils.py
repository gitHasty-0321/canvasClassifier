# Importing necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F


class EMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 32*14*14

            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),  # 256*7*7

            nn.Flatten(),
            nn.Linear(256*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 26)
        )

    def forward(self, images):
        return self.network(images)


def transform_image(image):  # convert all images into a similar size
    transforms = T.ToTensor()
    return transforms(image)


def load_checkpoint(filepath):  # loading the pretrained weights
    model = EMNIST()
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    model.eval()

    return model

# given the kind of input, chooses the model to predict on, returns a numpy array


def get_prediction(image_tensor):
    image_tensor = image_tensor.unsqueeze_(0)
    model = load_checkpoint(PATH)
    outputs = model(image_tensor)
    # _, predicted = torch.max(outputs.data, 1)
    return outputs.squeeze().detach().numpy()


PATH = 'app/model.pth'
