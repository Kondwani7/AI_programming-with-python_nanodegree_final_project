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

def train_network(model, epochs, learning_rate, trainloader, validloader, gpu):
    """
    Descripiton:
        trains a model based of given data
    Args:
        model: the model
        epochs: the number of times it is trained to achieve a lower gradient descent
        learning_rate: proposed rate at which the model descends
        trainloader: train data
        validloader: validation data
        gpu: environment used to handle large data
    Returns:
        model - performance of the model
        criterion - measure used to evaluate model performance
    """
    print("Training network ... epochs: {}, learning_rate: {}, gpu used for training: {}".format(epochs, learning_rate, gpu))
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    model.to(device)
    #training our model
    steps = 0
    print_every = 10
    train_loss = 0
    #same as the notebook in the visualizer file
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0

                model.eval()

                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate validation acc
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train loss: {train_loss/print_every:.3f}, "
                      f"Valid loss: {valid_loss/len(validloader):.3f}, "
                      f"Valid accuracy: {valid_accuracy/len(validloader):.3f}")

                train_loss = 0

                model.train()
                
    print("Finished training network")
    
    return model, criterion