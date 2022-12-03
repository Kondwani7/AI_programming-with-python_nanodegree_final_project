import torch
import time
import numpy as np
import json
import sys
import argparse

from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image
parser = argparse.ArgumentParser()
args = parser.parse_args()

def load_model():
    """
        Description:
            This fuction loads a given model info with it's state and classifier
        Returns:
            the model
    """
    model_info = torch.load(args.model_checkpoint)
    model = model_info['model']
    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['state_dict'])
    
    return model

def process_image(image):
    """
    Description:
        -processes an image so it can be read by a model
    Args:
        image
    Returns:
        image being processed as a numpy array
    """
    im = Image.open(image)
    width, height = im.size
    picture_coords = [width, height]
    max_span = max(picture_coords)
    max_element = picture_coords.index(max_span)
    if (max_element == 0):
        min_element = 1
    else:
        min_element = 0
    aspect_ratio=picture_coords[max_element]/picture_coords[min_element]
    new_picture_coords = [0,0]
    new_picture_coords[min_element] = 256
    new_picture_coords[max_element] = int(256 * aspect_ratio)
    im = im.resize(new_picture_coords)   
    width, height = new_picture_coords
    left = (width - 244)/2
    top = (height - 244)/2
    right = (width + 244)/2
    bottom = (height + 244)/2
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)
    np_image = np_image.astype('float64')
    np_image = np_image / [255,255,255]
    np_image = (np_image - [0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def classify_image(image_path, topk=5):
    """
    Description:
        classifies the probability an image belonging to a certain class
        based on the params of a model

    Args:
        image_path
        topk - Defaults to 5.
    Returns:
        results - accuracy(probability) of the model
    """
    topk=int(topk)
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        model = load_model()
        if (args.gpu):
           image = image.cuda()
           model = model.cuda()
        else:
            image = image.cpu()
            model = model.cpu()
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topk)
        probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
        results = zip(probs,classes)
        
        return results
    
def read_categories():
        """
        Description:
            read the categories of images
        Returns:
            jfile - json file of the categories otherwise if not found return None
        """
        if (args.category_names is not None):
            cat_file = args.category_names
            jfile = json.loads(open(cat_file).read())
            return jfile
        return None