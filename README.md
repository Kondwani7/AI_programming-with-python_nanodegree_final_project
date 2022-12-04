# Project: AI Programming with Python Nanodegree
![Interface](https://napaanalytics.com/wp-content/uploads/2020/04/Napa-Data-Engineering-Image.jpg)
## Table of Contents
- Summary
- considerations
- image classifier
- visualisation Notebok
- helper functions
- conclusion
### summary of project
This repo contains the submission of the final project for the AI for programming Nanodegree offered by [udacity](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089) sponosered by AWS. It is a neural network trained to classify flower species names based on the images and categories provided by the data set.
### Consideration
- Note that to data restrictions github places on pushing data, I could not upload the data used to predict the flowers. Use this link to access the [data](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) 
### image classifier
contains the visualize classifier and helper functions. The visualize classifier folder contains the notebook used to run the model. The helper functions help to deploy the model in a local workspace.
### visualize classifer
- where the [notebook](https://github.com/Kondwani7/AI_programming-with-python_nanodegree_final_project/blob/main/imageClassifer/visualize_classifier/Image%20Classifier%20Project.ipynb) resides.
- what models was used to train the function and its architechure
- the displaying the image model prediction performance
- Notably the model used the pretrained vgg16 pretrained network with my own custom classifier
- it 78.2% accuracy on the test set.
### helper functions
- [cat_json_name](https://github.com/Kondwani7/AI_programming-with-python_nanodegree_final_project/blob/main/cat_to_name.json): contained the a json file of all the names of the 102 flower categories
- the [data_manager.py](https://github.com/Kondwani7/AI_programming-with-python_nanodegree_final_project/blob/main/imageClassifer/helper_functions/data_manager.py) and [model_manager.py](https://github.com/Kondwani7/AI_programming-with-python_nanodegree_final_project/blob/main/imageClassifer/helper_functions/model_manager.py): contained the functions on how to process the images and data, how to construct, train and test the model, and how to return the model performance/results
- [train.py](https://github.com/Kondwani7/AI_programming-with-python_nanodegree_final_project/blob/main/imageClassifer/helper_functions/train.py): used to actually build, train and evaluate the model
- [predict.py](https://github.com/Kondwani7/AI_programming-with-python_nanodegree_final_project/blob/main/imageClassifer/helper_functions/predict.py): used to test the model based on a real image from our dataset, returning the probabilities of a flower image belowing to a certain class
### Conclusion
- using pretrained models is advantageous in regards to leveraging a model architecture that has worked on other unsupervised learning tasks
- writing functions in predict.py and train.py helps a machine learning engineer get out of a notebook, and test deploying a model in a potential production setting.
- Ideally, I would not run this in a local enviroment because of memory and RAM constraints.