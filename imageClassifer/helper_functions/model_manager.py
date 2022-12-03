import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def build_network(architecture, hidden_units):
    """
    Description:
        This function builds a models architectures based on specific parameters and hidden units
    Args:
        architecture: params and activation functions in the model
        hidden_units: layers used in the model
    Returns:
        model - params of the model
    """
    print("Building the network ... architecture: {}, hidden_units: {}".format(architecture, hidden_units))
    #different pretrained models that can be used and their input units
    if architecture =='vgg16':
        model = models.vgg16(pretrained = True)
        input_units = 25088
    elif architecture =='vgg13':
        model = models.vgg13(pretrained = True)
        input_units = 25088
    elif architecture =='alexnet':
        model = models.alexnet(pretrained = True)
        input_units = 9216
        
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(
              nn.Linear(input_units, hidden_units),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(hidden_units, 256),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(256, 102),
              nn.LogSoftmax(dim = 1)
            )
    
    model.classifier = classifier
    
    print("Completed building the neural network")
    
    return model