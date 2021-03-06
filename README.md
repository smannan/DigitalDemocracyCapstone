# DigitalDemocracyCapstone

This project identifies transition points from legislative transcripts. 

## Training models

Navigate to code/train and run either TEMP_generate_production_model.py or TEMP_generate_testing_model.py

## Generating predictions

Navigate to code/predict and run either TEMP_predict_from_production_model.py or TEMP_predict_from_testing_model.py

## Code

This directory contains all source code for the project. 
Run either train.create_production_model then predict.predict OR train.create_testing_model
then predict.evaluate_withheld_transcripts.

### Train

Contains code to extract features, split data into train/test datasets, and lastly trains and creates
a naive bayes model. The model will be saved to a pickle file in the model/ directory discussed below
and can be used for predictions. 

model.py contains two functions - create_production_model and create_testing_model.

The production model will use the entire dataset to create a model and is the one that should be used
in production environments.

The testing model will withold a variable number of transcripts specified by the user for testing and
verification purposes. 

Both functions require an original raw filename containing the raw transcripts, original upleveled 
filename containing upleveled transcripts, and the desired model type ('NN' for neural network or 'NB'
for naive bayes). 

create_testing_model takes one additional argument, withheld_video_ids, which is a list of video ids
to withhold from the model.

### Predict

Contains code to predict which statements signal bill transitions or not, given a new transcript. 
There are two functions, predict and evaluate_withheld_transcripts.

Both functions require a list of dictionaries and a model type. It will output a list of dicionaries 
containing predicted transitions, suggested bill names, and utterance start and end times for the transcript.

## Data

The data directory contains original, cleaned, and training data. The cleaned data contains the processed raw and upleveled data. If the user is creating a testing model the cleaned data will not contain the user-specified withheld transcripts. The training directory contains the utterance data and label (transition or non-transition). Similarly, if the user is creating a testing model, a withheld training set with removed transcripts will be created instead.

## Model

The naive bayes model and word vectorizer are saved to a pickle files whereas the neural network model is saved to a single HDF5 file. These models are created in the training portion and used in the prediction portion.

