import torch
from torchvision import datasets, transforms, models
from PIL import Image

def load_data(path):
    """
    Description:
        Loads the train, validation and test data from a given folder for model training
    Args:
        path - directory where the data resides
    Returns:
        the train data, and the loaders for the train, validation and test loaders
    """
    print("Loading and preprocessing data from {} ...".format(path))
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    
    # Define transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.RandomRotation(50),
                                                  transforms.RandomResizedCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    print("Finished loading and preprocessing data.")
    
    return train_data, trainloader, validloader, testloader

def process_image(image):
    """
    Description:
        processes (resize, crop and normalize)an image for the model as a numpy array
    Args:
        image - PIL imge
    Returns:
        a numpy array of transformed images, split as train, validation and test
    """
    image = Image.open(image)
    
    image_transform = transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])
    
    return image_transform(image)