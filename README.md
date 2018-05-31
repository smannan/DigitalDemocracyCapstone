# DigitalDemocracyCapstone

This project identifies transition points from legislative transcripts. 

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

Contains the data necessary for the pipeline.

## Model

Contains models.

