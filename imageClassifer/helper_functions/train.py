import argparse
from data_manager import load_data
import model_manager
parser = argparse.ArgumentParser(description="Training a neural network on our dataset")
parser.add_argument('data_directory', help='the path to where our data for training resides')
parser.add_argument('--save_dir', help='Path to where our checkpoint should be saved')
parser.add_argument('--arch', help='Network architecture (default model \'vgg16\')')
parser.add_argument('--learning_rate', help='Learning rate')
parser.add_argument('--hidden_units', help='Number of hidden units')
parser.add_argument('--epochs', help='Number of epochs')
parser.add_argument('--gpu', help='Use GPU for training', action='store_true')

args = parser.parse_args()

save_dir = '' if args.save_dir is None else args.save_dir
network_architecture = 'vgg16' if args.arch is None else args.arch
learning_rate = 0.0025 if args.learning_rate is None else int(args.learning_rate)
hidden_units = 512 if args.hidden_units is None else float(args.hidden_units)
epochs = 5 if args.epochs is None else int(args.epochs)
gpu = False if args.gpu is None else True
#data split to train, validation and test sets
train_data, trainloader, validloader, testloader = load_data(args.data_directory)
#build our model
model = model_manager.build_network(network_architecture, hidden_units)
model.class_to_idx = train_data.class_to_idx
#train our model
model, criterion = model_manager.train_network(model, epochs, learning_rate, trainloader, validloader, gpu)
#evaluate our model
model_manager.evaluate_model(model, testloader, criterion, gpu)
#save our model
model_manager.save_model(model, network_architecture, hidden_units, epochs, learning_rate, save_dir)
