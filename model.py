import torch
import torchvision.models as models

torch.manual_seed(50)  

# Define the ResNet50 model
def get_resnet50(pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    return model

# If running this file directly, it will print the model architecture
if __name__ == "__main__":
    model = get_resnet50()
    print(model)