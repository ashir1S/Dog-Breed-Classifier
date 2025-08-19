import torch
import torchvision
from torch import nn

def create_resnet50_model(num_classes: int = 157, seed: int = 42):
    """Creates a ResNet50 model with a custom classifier head.

    Args:
        num_classes (int, optional): number of classes in the classifier head.
            Defaults to 157.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): ResNet50 model.
    """
    # Load a pre-trained ResNet50 model
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

    # Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    # Get the number of input features for the final classification layer
    num_ftrs = model.fc.in_features

    # Replace the final fully connected layer (the classifier)
    torch.manual_seed(seed)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features=num_ftrs, out_features=num_classes)
    )

    return model
