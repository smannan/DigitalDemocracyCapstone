# DigitalDemocracyCapstone

## Model Training

This folder contains the code to train our machine learning models. It contains notebooks to process raw data and extract features, combine raw data with upleveled data, create training and testing datasets, then train either a Naive Bayes or Neural Network model.

The output of model training will be a pickle file containing the Naive Bayes model and text vectorizer or a .hdf5
file containing the weights of the neural network model.

Run model training before pipeline to traing the model first.


## Pipeline

This folder contains two notebooks: Raw_Data_Processing and Make_Predictions. It also contains a python file Predict_Transitions that combines the code in the two python notebooks.

The code in pipeline expects a list of dictionaries containing utterance start time, end time, video id, and text.
The output will then be a list of dictionaries containing transition utterances. The python file 
Predict_Transitions.py contains a function, predict, that will output transition utterances given a list of dictionaries.

