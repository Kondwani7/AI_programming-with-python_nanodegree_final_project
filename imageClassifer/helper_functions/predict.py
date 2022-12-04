import argparse
import json
import model_manager
import data_manager
#arguments
parser = argparse.ArgumentParser(description='Predict the probabiility of an image of a flower belonging to one of our categories.')
parser.add_argument('image_path', help='Path to image')
parser.add_argument('checkpoint', help='Checkpoint of a neural network')
parser.add_argument('--top_k', help='Return top k classes')
parser.add_argument('--category_names', help='Map categories to real  flower names')
parser.add_argument('--gpu', help='Use GPU for inference', action='store_true')

args = parser.parse_args()
#assigning topk, category name and gpu sources
top_k = 1 if args.top_k is None else int(args.top_k)
category_names = "cat_to_name.json" if args.category_names is None else args.category_names
gpu = False if args.gpu is None else True
#loading our model
model = model_manager.load_model(args.checkpoint)
print(model)
#prediction and probability
probs, predict_classes = model_manager.predict(data_manager.process_image(args.image_path), model, top_k)
#open cateorgy from json file of categories
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

classes = []

for predict_class in predict_classes:
    classes.append(cat_to_name[predict_class])
#print probability and classes
print(probs)
print(classes)